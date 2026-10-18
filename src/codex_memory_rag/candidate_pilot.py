"""Local candidate presentation. This module never reads files or loads models."""
import re
from pathlib import PurePosixPath

from .schemas import CmrError


CANDIDATE_WARNINGS = ("CANDIDATE_PILOT", "SUPPORT_NOT_CHECKED", "完整支持未检查")
CANDIDATE_FIELDS = {"retrieval_role": "related_candidate", "support_status": "not_checked"}


def candidate_warnings(config):
    return list(CANDIDATE_WARNINGS) if config.candidate_pilot.enabled else []


def is_candidate_payload(data):
    if not isinstance(data, dict):
        return False
    warnings = data.get("warnings", [])
    if isinstance(warnings, list) and "CANDIDATE_PILOT" in warnings:
        return True
    for name in ("items", "results"):
        entries = data.get(name, [])
        if isinstance(entries, list) and any(
            isinstance(x, dict) and
            (x.get("retrieval_role") == "related_candidate" or x.get("support_status") == "not_checked")
            for x in entries
        ):
            return True
    return False


def require_candidate_response(config, data, collection=None):
    """Fail closed when a candidate client reaches an older/different daemon."""
    if not config.candidate_pilot.enabled and not is_candidate_payload(data):
        return
    if not isinstance(data, dict) or not isinstance(data.get("warnings"), list):
        raise CmrError("CANDIDATE_CONTRACT_MISMATCH")
    if any(warning not in data["warnings"] for warning in CANDIDATE_WARNINGS):
        raise CmrError("CANDIDATE_CONTRACT_MISMATCH")
    if collection is not None:
        entries = data.get(collection)
        if not isinstance(entries, list) or any(
            not isinstance(x, dict) or
            any(x.get(key) != value for key, value in CANDIDATE_FIELDS.items())
            for x in entries
        ):
            raise CmrError("CANDIDATE_CONTRACT_MISMATCH")


def _item_with_path(config, project_id, item):
    if not isinstance(item, dict):
        raise CmrError("RETRIEVAL_RESPONSE_INVALID")
    sources = [s for s in config.sources if s.enabled and s.allow_cloud_context
               and s.project_id == project_id and s.source_id == item.get("source_id")
               and s.worktree_id == item.get("worktree_id")]
    if item.get("project_id") != project_id or len(sources) != 1:
        raise CmrError("SCOPE_DENIED")
    for key in ("chunk_id", "file_sha256"):
        if not isinstance(item.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", item[key]):
            raise CmrError("RETRIEVAL_RESPONSE_INVALID")
    start, end = item.get("line_start"), item.get("line_end")
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        raise CmrError("RETRIEVAL_RESPONSE_INVALID")
    relative = item.get("relative_path")
    if not isinstance(relative, str) or not relative or "\0" in relative:
        raise CmrError("SCOPE_DENIED")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative or relative == ".":
        raise CmrError("SCOPE_DENIED")
    return item | {"path": str(sources[0].root / path)}


def search_cli_output(config, request, data):
    require_candidate_response(config, data, "results")
    if not is_candidate_payload(data):
        return data
    if (not isinstance(data.get("generation"), str) or not data["generation"].strip()
            or type(data.get("table_version")) is not int or data["table_version"] < 0
            or not isinstance(data.get("mode"), str) or not data["mode"].strip()
            or type(data.get("truncated")) is not bool
            or any(not isinstance(warning, str) for warning in data["warnings"])):
        raise CmrError("RETRIEVAL_RESPONSE_INVALID")
    return data | {"results": [
        _item_with_path(config, request.project_id, item) for item in data["results"]
    ]}


def context_cli_output(config, request, data):
    require_candidate_response(config, data, "items")
    if (not isinstance(data, dict) or not isinstance(data.get("items"), list)
            or not isinstance(data.get("generation"), str)
            or type(data.get("table_version")) is not int
            or type(data.get("truncated")) is not bool):
        raise CmrError("RETRIEVAL_RESPONSE_INVALID")
    if "warnings" in data and (not isinstance(data["warnings"], list)
                              or any(not isinstance(x, str) for x in data["warnings"])):
        raise CmrError("RETRIEVAL_RESPONSE_INVALID")
    items = [_item_with_path(config, request.project_id, item) for item in data["items"]]
    keys = [item["chunk_id"] for item in items]
    if keys != request.chunk_ids[:len(keys)]:
        raise CmrError("SCOPE_DENIED")
    if any(not isinstance(item.get("text"), str) for item in items):
        raise CmrError("RETRIEVAL_RESPONSE_INVALID")
    partial = data["truncated"] or len(keys) != len(request.chunk_ids) or any(
        not item["text"] for item in items
    )
    result = data | {"items": items, "readback_status": "partial" if partial else "complete"}
    if partial:
        result["warnings"] = list(data.get("warnings", [])) + ["原文回读未完成"]
    return result, 3 if partial else 0
