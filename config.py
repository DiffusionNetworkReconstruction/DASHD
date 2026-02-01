from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class AppConfig:
    log_level: str = "INFO"


def load_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return AppConfig(log_level=str(data.get("log_level", "INFO")).upper())
