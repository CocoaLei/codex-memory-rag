import tomllib
from pathlib import Path

from .schemas import AppConfig, CmrError


def load_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def registered_sources(config: AppConfig, project: str, kind: str | None = None):
    sources = [s for s in config.sources if s.project_id == project and s.enabled]
    if not sources:
        raise CmrError("SOURCE_NOT_CONFIGURED")
    return [s for s in sources if s.allow_cloud_context and (kind is None or s.kind == kind)]
