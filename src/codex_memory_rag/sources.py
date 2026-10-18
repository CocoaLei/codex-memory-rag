"""Allowlisted, descriptor-based source access and source-derived manifests."""
import fnmatch
import hashlib
import os
import re
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pathspec
import yaml

from .schemas import (CmrError, CorpusManifest, ScanResult, SourceConfig,
                      SourceDocument, identity_hash)

MAX_FILE = 2 * 1024 * 1024
STATE_RULE_VERSION = "state-v2"
EXCLUDED_PARTS = {".git", ".obsidian", "node_modules", ".venv", "venv", "__pycache__",
                  "build", "dist", ".cache", ".sites", ".playwright-cli", "cmr-runtime",
                  "shell_snapshots"}
ALLOWED_SUFFIXES = {".md", ".py", ".ts", ".tsx", ".js", ".jsx", ".kt", ".java",
                    ".json", ".yaml", ".yml", ".xml", ".toml", ".txt"}
SECRET = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-|hf_)[A-Za-z0-9_-]{20,}|(?i:(?:api[_-]?key|password|secret|token))\s*[:=]\s*[\"'][A-Za-z0-9_+/=.-]{16,}[\"']")


def _relative(path: str) -> list[str]:
    parts = path.split("/")
    if not path or path.startswith("/") or "://" in path or any(x in ("", ".", "..") for x in parts):
        raise CmrError("SCOPE_DENIED")
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeError:
        raise CmrError("SCOPE_DENIED") from None
    return parts


@contextmanager
def _held_root(root: Path):
    """Anchor enumeration and reads to the same trusted root ancestry."""
    fds, chain, bound = [], [], False
    try:
        try:
            if not root.is_absolute() or ".." in root.parts:
                raise CmrError("STALE_SOURCE")
            original = root.lstat()
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            fd = os.open("/", flags); fds.append(fd)
            for component in root.parts[1:]:
                child = os.open(component, flags, dir_fd=fd)
                fds.append(child); chain.append((fd, component, child)); fd = child
            actual = os.fstat(fd)
            if not stat.S_ISDIR(original.st_mode) or (original.st_dev, original.st_ino) != (actual.st_dev, actual.st_ino):
                raise CmrError("STALE_SOURCE")
            bound = True
        except OSError:
            raise CmrError("STALE_SOURCE") from None
        yield fd
    finally:
        try:
            if bound:
                try:
                    for parent, name, held in chain:
                        linked = os.stat(name, dir_fd=parent, follow_symlinks=False); actual = os.fstat(held)
                        if stat.S_ISLNK(linked.st_mode) or (linked.st_dev, linked.st_ino) != (actual.st_dev, actual.st_ino):
                            raise CmrError("STALE_SOURCE")
                except OSError:
                    raise CmrError("STALE_SOURCE") from None
        finally:
            for descriptor in reversed(fds): os.close(descriptor)


def _walk_source(root):
    with _held_root(root) as root_fd:
        for directory, dirs, files, descriptor in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_PARTS and not stat.S_ISLNK(os.stat(d, dir_fd=descriptor, follow_symlinks=False).st_mode))
            prefix = "" if directory == "." else directory.removeprefix("./") + "/"
            yield prefix, files


def secure_read(root: Path, relative: str) -> bytes:
    """Descriptor read, bounded bytes, unchanged file and still-linked roots."""
    parts = _relative(relative)
    fds, chain = [], []
    try:
        with _held_root(root) as fd:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            for part in parts[:-1]:
                child = os.open(part, flags, dir_fd=fd)
                fds.append(child); chain.append((fd, part, child)); fd = child
            child = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            fds.append(child); chain.append((fd, parts[-1], child))
            before = os.fstat(child)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE:
                raise CmrError("SCOPE_DENIED")
            data = bytearray()
            while block := os.read(child, min(65536, MAX_FILE + 1 - len(data))):
                data.extend(block)
                if len(data) > MAX_FILE: raise CmrError("SCOPE_DENIED")
            after = os.fstat(child)
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise CmrError("SOURCE_CHANGED")
            for parent, name, held in chain:
                linked = os.stat(name, dir_fd=parent, follow_symlinks=False); actual = os.fstat(held)
                if stat.S_ISLNK(linked.st_mode) or (linked.st_dev,linked.st_ino) != (actual.st_dev,actual.st_ino):
                    raise CmrError("SOURCE_CHANGED")
            return bytes(data)
    except CmrError:
        raise
    except FileNotFoundError:
        raise CmrError("SOURCE_MISSING") from None
    except (OSError, UnicodeError):
        raise CmrError("SCOPE_DENIED") from None
    finally:
        for descriptor in reversed(fds): os.close(descriptor)


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) or PurePosixPath(path).match(p) for p in patterns)


def allowed_path(source: SourceConfig, relative: str) -> bool:
    parts = _relative(relative)
    absolute_parts = (source.root / relative).parts
    if ".codex" in absolute_parts:
        index = absolute_parts.index(".codex")
        if len(absolute_parts) > index + 1 and absolute_parts[index + 1] in {"sessions", "auth.json", "memories", "backups"}:
            return False
    name = parts[-1].lower()
    if not source.enabled or not source.allow_cloud_context:
        return False
    if any(p in EXCLUDED_PARTS for p in parts) or name.startswith(".env"):
        return False
    if name in {"auth.json", "config.toml", "id_rsa", "id_ed25519"} or Path(name).suffix in {".pem", ".key", ".p12", ".jks", ".keystore", ".lock"}:
        return False
    if Path(name).suffix not in ALLOWED_SUFFIXES or not _matches(relative, source.include) or _matches(relative, source.exclude):
        return False
    for depth in range(len(parts)):
        prefix = parts[:depth]
        subpath = "/".join(parts[depth:])
        for ignore_name in [".gitignore", ".memoryignore"]:
            rel = "/".join(prefix + [ignore_name])
            try:
                rules = secure_read(source.root, rel).decode("utf-8-sig")
            except CmrError as exc:
                if exc.code == "SOURCE_MISSING":
                    continue
                # An unreadable source-wide policy is not an instruction to
                # exclude every file and delete its previous indexed rows.
                raise CmrError("STALE_SOURCE") from None
            except UnicodeError:
                raise CmrError("STALE_SOURCE") from None
            try:
                if pathspec.GitIgnoreSpec.from_lines(rules.splitlines()).match_file(subpath):
                    return False
            except ValueError:
                raise CmrError("STALE_SOURCE") from None
    return True


def git_state(source: SourceConfig) -> dict | None:
    if source.kind != "code":
        return None
    executable = shutil.which("git")
    if not executable or not Path(executable).is_absolute():
        raise CmrError("STALE_SOURCE")
    def run(*args):
        # Absolute executable + close_fds=False selects posix_spawn on macOS
        # Python3.12. All owned fds are non-inheritable (PEP446). Avoid fork
        # after Torch/tokenizer/native worker threads have been initialized.
        try:
            return subprocess.run([executable, "-C", str(source.root), *args], stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=3, close_fds=False)
        except (OSError, subprocess.TimeoutExpired):
            raise CmrError("STALE_SOURCE") from None
    def git(*args):
        result = run(*args)
        if result.returncode:
            raise CmrError("STALE_SOURCE")
        return result.stdout.strip()
    values = git("rev-parse", "HEAD", "--absolute-git-dir", "--git-common-dir", "--git-path", "HEAD").splitlines()
    if len(values) != 4:
        raise CmrError("STALE_SOURCE")
    head, gitdir, common_dir, head_path = values
    branch_result = run("symbolic-ref", "-q", "HEAD")
    if branch_result.returncode not in (0, 1):
        raise CmrError("STALE_SOURCE")
    return {"head": head, "branch": branch_result.stdout.strip() or "detached",
            "gitdir": gitdir,
            "common_dir": str((source.root / common_dir).resolve()),
            "head_path": str((source.root / head_path).resolve()),
            "worktree": str(source.root)}


def read_allowed_file(source: SourceConfig, relative_path: str) -> SourceDocument:
    if not allowed_path(source, relative_path):
        raise CmrError("SCOPE_DENIED")
    before_git = git_state(source)
    raw = secure_read(source.root, relative_path)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        raise CmrError("SCOPE_DENIED") from None
    if "\x00" in text or SECRET.search(text):
        raise CmrError("SCOPE_DENIED")
    metadata = {}
    if text.startswith("---") and Path(relative_path).suffix == ".md":
        match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, re.S)
        if match:
            try:
                metadata = yaml.safe_load(match[1]) or {}
            except yaml.YAMLError:
                raise CmrError("SCOPE_DENIED") from None
            if not isinstance(metadata, dict):
                raise CmrError("SCOPE_DENIED")
    if source.kind == "memory" and source.code_project and metadata.get("code_project") != source.code_project:
        raise CmrError("SCOPE_DENIED")
    after_git = git_state(source)
    if before_git != after_git:
        raise CmrError("STALE_SOURCE")
    return SourceDocument(source=source, relative_path=relative_path, text=text,
                          file_sha256=hashlib.sha256(raw).hexdigest(), metadata=metadata, git_state=after_git)


def derive_retrieval_state(document: SourceDocument) -> tuple[str, list[str]]:
    if document.source.kind != "memory":
        return "source_current", []
    meta = document.metadata
    value = meta.get("status")
    reasons = []
    if value in {"deprecated", "superseded", "archived", "rejected"}:
        reasons.append(str(value))
    until = meta.get("valid_until")
    unresolved_expiry = False
    if until:
        try:
            date = datetime.fromisoformat(str(until).replace("Z", "+00:00"))
            if date.tzinfo and date < datetime.now(timezone.utc):
                reasons.append("expired")
            elif not date.tzinfo:
                unresolved_expiry = True
        except (ValueError, TypeError):
            unresolved_expiry = True
    archived = meta.get("type") == "archive" or any(p in document.source.archive_dirs for p in PurePosixPath(document.relative_path).parts)
    if archived:
        reasons.append("archive_location_or_type")
        if value == "active":
            reasons.append("status_conflict")
    if reasons:
        return "historical", reasons
    if unresolved_expiry:
        return "unverified", ["invalid_or_timezone_unspecified_expiry"]
    if value == "active":
        return "eligible", []
    return "unverified", [str(value) if value is not None else "missing_status"]


def explicit_replacements(document: SourceDocument, sources: list[SourceConfig], candidate_cache=None) -> list[str]:
    """Resolve explicit same-project identities, never mtime or fuzzy basenames."""
    from .schemas import source_key
    if document.source.kind != "memory":
        return []
    candidates = []
    for source in sources:
        if source.project_id != document.source.project_id or source.kind != "memory":
            continue
        cache_key = (None, "replacement_documents", source.source_id, source.project_id)
        documents = candidate_cache.get(cache_key) if candidate_cache is not None else None
        if documents is None:
            scan = scan_source(source)
            if not scan.accessible or scan.failed:
                raise CmrError("STALE_SOURCE")
            documents = [(path, read_allowed_file(source, path)) for path in scan.allowed]
            if candidate_cache is not None:
                candidate_cache[cache_key] = documents
        for path, newer in documents:
            if source.source_id == document.source.source_id and path == document.relative_path:
                continue
            if derive_retrieval_state(newer)[0] != "eligible":
                continue
            references = newer.metadata.get("supersedes", [])
            references = [references] if isinstance(references, str) else references
            if not isinstance(references, list):
                continue
            key = source_key(document.source.project_id, document.source.source_id, None, document.relative_path)
            explicit = {key, document.source.source_id + ":" + document.relative_path}
            if source.source_id == document.source.source_id:
                explicit.add(document.relative_path)
            if any(isinstance(ref, str) and ref in explicit for ref in references):
                candidates.append(source.source_id + ":" + path)
    return candidates


def scan_source(source: SourceConfig) -> ScanResult:
    steps = scan_source_steps(source)
    while True:
        try:
            next(steps)
        except StopIteration as completed:
            return completed.value


def scan_source_steps(source: SourceConfig):
    """Incremental variant retaining all scope, readability, and root checks."""
    result = ScanResult()
    try:
        for prefix, files in _walk_source(source.root):
            for name in sorted(files):
                path = prefix + name
                if Path(name).suffix not in ALLOWED_SUFFIXES and not name.startswith(".env"):
                    continue
                try:
                    read_allowed_file(source, path)
                    result.allowed.append(path)
                except CmrError as exc:
                    target = result.rejected if exc.code == "SCOPE_DENIED" else result.failed
                    target[path] = exc.code
                yield
    except (CmrError, OSError):
        result.accessible = False
        result.failed["_root"] = "STALE_SOURCE"
    result.allowed.sort()
    return result


def build_corpus_manifest(sources: list[SourceConfig]) -> CorpusManifest:
    files, coverage, states, policies = [], {}, {}, []
    for source in sources:
        scan = scan_source(source)
        if not scan.accessible or scan.failed:
            raise CmrError("STALE_SOURCE")
        coverage[source.source_id] = {"allowed": len(scan.allowed), "rejected": len(scan.rejected), "failed": len(scan.failed)}
        states[source.source_id] = git_state(source)
        # Ignore policy itself is part of the frozen corpus, including rules
        # whose current targets happen to be absent.
        for prefix, names in _walk_source(source.root):
            for name in sorted(set(names) & {".gitignore", ".memoryignore"}):
                relative = prefix + name
                policies.append([source.source_id, relative, hashlib.sha256(secure_read(source.root, relative)).hexdigest()])
        for path in scan.allowed:
            document = read_allowed_file(source, path)
            files.append({"project_id": source.project_id, "source_id": source.source_id,
                          "worktree_id": source.worktree_id, "relative_path": path,
                          "file_sha256": document.file_sha256})
    policy = identity_hash([[s.model_dump(mode="json") for s in sources], sorted(policies), STATE_RULE_VERSION])
    files.sort(key=lambda f: (f["project_id"], f["source_id"], f["relative_path"]))
    data = {"files": files, "policy_hash": policy, "git_states": states, "coverage": coverage}
    return CorpusManifest(**data, corpus_manifest_hash=identity_hash(data))
