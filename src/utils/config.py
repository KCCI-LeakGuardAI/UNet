from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return value


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = Path(base or Path.cwd()) / result
    return result.resolve()

