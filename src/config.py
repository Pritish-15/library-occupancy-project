"""Load project configuration from config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def data_path(key: str) -> Path:
    cfg = load_config()
    return ROOT / cfg["paths"][key]


def ensure_dirs() -> None:
    cfg = load_config()
    for key in ("raw_data", "processed_data"):
        path = ROOT / cfg["paths"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    project_path("notebooks").mkdir(parents=True, exist_ok=True)
    project_path("docs").mkdir(parents=True, exist_ok=True)
    project_path("tests").mkdir(parents=True, exist_ok=True)
    calendar_file = ROOT / cfg["paths"].get("academic_calendar", "data/calendar/academic_calendar.json")
    calendar_file.parent.mkdir(parents=True, exist_ok=True)
    project_path("data", "uploads").mkdir(parents=True, exist_ok=True)
