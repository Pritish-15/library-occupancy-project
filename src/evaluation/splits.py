"""Chronological train / validation / test splits."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


def chronological_split(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    ratios = cfg["splits"]
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])
    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    return train, val, test
