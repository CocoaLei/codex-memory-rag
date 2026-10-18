"""Bind actual interpreter and installed bytes, without importing any model."""
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import sys
import sysconfig
import time
from pathlib import Path

MAX_FILES = 100000
MAX_BYTES = 50 * 1024 ** 3
MAX_SECONDS = 120
VARIABLES = ("PYTHONPATH", "PYTHONHASHSEED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
             "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS")


class EnvironmentDrift(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def environment_spec():
    sites = sorted({str(Path(value).resolve()) for value in sys.path
                    if value and Path(value).name in {"site-packages", "dist-packages"}})
    sites = sorted(set(sites + [str(Path(sysconfig.get_path("purelib")).resolve()),
                               str(Path(sysconfig.get_path("platlib")).resolve())]))
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    return {"python": os.path.abspath(sys.executable), "venv": os.path.abspath(sys.prefix),
            "stdlib": sysconfig.get_path("stdlib"), "site_paths": sites,
            "sys_path": [os.path.abspath(value or os.getcwd()) for value in sys.path],
            "allowed_roots": sorted(set([os.path.realpath(sys.prefix),
                                         os.path.realpath(sys.base_prefix)] + sites
                                         + [os.path.realpath(value or os.getcwd())
                                            for value in sys.path if Path(value or os.getcwd()).is_dir()])),
            "interpreter_related": [str(library)]}


def capture_environment(spec=None, *, metadata_only=False):
    spec = spec or environment_spec()
    allowed = [Path(root).resolve() for root in spec["allowed_roots"]]
    budget = {"files": 0, "bytes": 0}
    deadline = time.monotonic() + MAX_SECONDS

    def check_budget():
        if (budget["files"] > MAX_FILES or budget["bytes"] > MAX_BYTES
                or time.monotonic() > deadline):
            raise EnvironmentDrift("ENVIRONMENT_SCAN_BUDGET")

    def identity(path, ancestors=(), trusted_parent=False):
        path = Path(path)
        check_budget()
        if not path.exists() and not path.is_symlink():
            return {"exists": False}
        try:
            info = path.lstat()
            # For metadata only, the enclosing traversal already resolved and
            # scope-checked the parent. Links and direct anchors never use this
            # fast path; complete byte scans retain their original resolver.
            if metadata_only and trusted_parent and stat.S_ISREG(info.st_mode):
                resolved = None
            else:
                resolved = path.resolve(strict=True)
                if not any(resolved.is_relative_to(root) for root in allowed):
                    raise EnvironmentDrift("ENVIRONMENT_SCOPE_UNSUPPORTED")
            item = {"exists": True, "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid}
            if stat.S_ISLNK(info.st_mode):
                if str(path) in ancestors or len(ancestors) >= 32:
                    raise EnvironmentDrift("ENVIRONMENT_SCOPE_UNSUPPORTED")
                target = os.readlink(path)
                linked = Path(os.path.abspath(path.parent / target))
                item.update(kind="symlink", target=target, resolved=str(resolved),
                            target_identity=identity(linked, ancestors + (str(path),)))
                if os.readlink(path) != target:
                    raise EnvironmentDrift("ENVIRONMENT_CHANGED")
            elif stat.S_ISREG(info.st_mode):
                if metadata_only:
                    budget["files"] += 1
                    item.update(kind="file", bytes=info.st_size, dev=info.st_dev,
                                ino=info.st_ino, mtime_ns=info.st_mtime_ns,
                                ctime_ns=info.st_ctime_ns)
                else:
                    checksum = hashlib.sha256()
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(fd, "rb") as stream:
                        initial = os.fstat(stream.fileno())
                        if (initial.st_dev, initial.st_ino) != (info.st_dev, info.st_ino):
                            raise EnvironmentDrift("ENVIRONMENT_CHANGED")
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            checksum.update(block)
                            budget["bytes"] += len(block)
                            check_budget()
                        final = os.fstat(stream.fileno())
                    after = path.lstat()
                    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                    if any(getattr(initial, key) != getattr(final, key)
                           or getattr(initial, key) != getattr(after, key) for key in fields):
                        raise EnvironmentDrift("ENVIRONMENT_CHANGED")
                    budget["files"] += 1
                    item.update(kind="file", bytes=final.st_size, sha256=checksum.hexdigest())
            elif stat.S_ISDIR(info.st_mode):
                item.update(kind="directory")
            else:
                raise EnvironmentDrift("ENVIRONMENT_SCOPE_UNSUPPORTED")
            return item
        except (OSError, RuntimeError) as exc:
            if isinstance(exc, EnvironmentDrift):
                raise
            raise EnvironmentDrift("ENVIRONMENT_CHANGED") from None

    def tree(root, providers_only=False):
        root = Path(root)
        members = {}

        def visit(path, ancestry, trusted_parent=False):
            directory = path.is_dir()
            real = path.resolve() if directory else None
            if directory and real in ancestry:
                raise EnvironmentDrift("ENVIRONMENT_SCOPE_UNSUPPORTED")
            initial_directory = path.lstat() if directory else None
            members[str(path.relative_to(root)) or "."] = identity(path, trusted_parent=trusted_parent)
            if directory:
                for child in sorted(path.iterdir()):
                    if path == Path(spec["stdlib"]) and child.name in {"site-packages", "dist-packages"}:
                        # Active package roots are inventoried separately. Do not
                        # bind another environment's inactive base packages.
                        continue
                    # Every directory is a potential namespace package. Bind
                    # its existence plus all loadable source/cache/native files.
                    # Non-code report bodies cannot act as FileFinder providers.
                    suffixes = (".py", ".pyc", ".pyo") + tuple(importlib.machinery.EXTENSION_SUFFIXES)
                    if providers_only and not child.is_dir() and not child.name.endswith(suffixes):
                        continue
                    visit(child, ancestry + (real,), trusted_parent=True)
                final_directory = path.lstat()
                # Detect ancestor substitution during enumeration without
                # rejecting ordinary non-importable output-file creation.
                if (initial_directory.st_dev, initial_directory.st_ino, initial_directory.st_mode) != (
                        final_directory.st_dev, final_directory.st_ino, final_directory.st_mode):
                    raise EnvironmentDrift("ENVIRONMENT_CHANGED")
        visit(root, ())
        return members

    try:
        distributions = sorted(
            [{"name": distribution.metadata["Name"], "version": distribution.version,
              "metadata_path": str(distribution._path)}
             for distribution in importlib.metadata.distributions(path=spec["site_paths"])],
            key=lambda item: (item["name"].lower(), item["metadata_path"]))
        actual_module = importlib.util.find_spec("codex_memory_rag")
        providers = {}
        separately_bound = {os.path.abspath(spec["stdlib"])} | {
            os.path.abspath(root) for root in spec["site_paths"]}
        for entry in spec["sys_path"]:
            path = Path(entry)
            absolute = os.path.abspath(path)
            if absolute in separately_bound:
                providers[absolute] = {"bound_by": "stdlib-or-active-site-tree"}
            elif path.is_dir():
                providers[absolute] = {"directory": tree(path, providers_only=True)}
            else:
                # zipimport supports an archive filename of any suffix; bind
                # absent entries too, so absent -> present is always drift.
                providers[absolute] = {"identity": identity(path)}
        captured = {
            "schema": "actual-python-metadata-v2" if metadata_only else "actual-python-environment-v2", "spec": spec,
            "effective_import_providers": providers,
            "python_runtime": {"version": sys.version, "implementation": platform.python_implementation(),
                               "cache_tag": sys.implementation.cache_tag, "architecture": platform.machine(),
                               "flags": list(sys.flags), "cpu_count": os.cpu_count(),
                               "base_prefix": sys.base_prefix, "prefix": sys.prefix,
                               "os_release": platform.release(),
                               "application_origin": actual_module.origin if actual_module else None},
            "variables": {name: os.environ.get(name) for name in VARIABLES},
            "interpreter": identity(spec["python"]),
            "pyvenv_cfg": identity(Path(spec["venv"]) / "pyvenv.cfg"),
            "interpreter_related": {path: identity(path) for path in spec.get("interpreter_related", [])},
            "stdlib": tree(spec["stdlib"]),
            "site_packages": {root: tree(root) for root in spec["site_paths"]},
            "distributions": distributions,
            "excluded": ["inactive stdlib top-level package directories; active site_paths captured separately",
                         "non-importable report data bytes in extra sys.path directories"],
            "cache_policy": "all existing stdlib/site/source bytecode bound; -B only forbids writes",
        }
        if not captured["interpreter"]["exists"]:
            raise EnvironmentDrift("ENVIRONMENT_CHANGED")
        check_budget()
        return captured
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EnvironmentDrift("ENVIRONMENT_CHANGED") from exc


def verify_environment(expected, spec=None):
    actual = capture_environment(spec)
    if digest(actual) != digest(expected):
        raise EnvironmentDrift("ENVIRONMENT_CHANGED")
    return {"environment_sha256": digest(actual), "distributions": len(actual["distributions"]),
            "bound_files": sum(item.get("kind") == "file" for item in actual["stdlib"].values())
            + sum(item.get("kind") == "file" for tree in actual["site_packages"].values()
                  for item in tree.values()), "verified": True}


def capture_metadata(spec=None):
    """Immutable identity inventory; content verification remains mandatory per group."""
    return capture_environment(spec, metadata_only=True)


def verify_metadata(expected, spec=None):
    actual = capture_metadata(spec)
    if digest(actual) != digest(expected):
        raise EnvironmentDrift("ENVIRONMENT_CHANGED")
    return {"inventory_sha256": digest(actual), "verified": True,
            "policy": "group content hashes plus immutable inode/ctime inventory"}
