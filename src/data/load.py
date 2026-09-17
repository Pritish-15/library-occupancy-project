"""Swappable raw-data loader. Replace generate/write_raw with a real CSV at the same path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import data_path
from src.data.generate import write_raw


def load_raw(path: Path | None = None, generate_if_missing: bool = True) -> pd.DataFrame:
    csv_path = path or data_path("raw_data")
    if not csv_path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(f"Raw occupancy file not found: {csv_path}")
        write_raw()
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df.sort_values(["timestamp", "library_id", "zone_id"]).reset_index(drop=True)
