from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetInfo:
    target_col_used: str
    timestamp_col_used: Optional[str]
    inferred_points_per_day: Optional[int]
    num_rows: int
    original_num_rows: int
    subset_start_row: int
    subset_max_rows: int


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        c
        for c in df.columns
        if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()
    ]
    return candidates[0] if candidates else None


def infer_points_per_day(df: pd.DataFrame, timestamp_col: Optional[str]) -> Optional[int]:
    if timestamp_col is None or timestamp_col not in df.columns:
        return None

    ts = pd.to_datetime(df[timestamp_col], errors="coerce", dayfirst=True)
    ts = ts.dropna()
    if len(ts) < 3:
        return None

    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    if len(diffs) == 0:
        return None

    step_minutes = float(diffs.median())
    if step_minutes <= 0:
        return None

    return int(round((24.0 * 60.0) / step_minutes))


def choose_target_column(df: pd.DataFrame, target_col: Optional[str]) -> str:
    if target_col:
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in columns: {list(df.columns)}")
        return target_col

    preferred = ["Electricity_load", "electricity_load", "load", "Load"]
    for c in preferred:
        if c in df.columns:
            return c

    numeric_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("No numeric column found in the dataset.")

    return numeric_cols[0]


def load_series(
    csv_path: str,
    target_col: Optional[str],
    start_row: int = 0,
    max_rows: int = 0,
) -> Tuple[np.ndarray, DatasetInfo]:
    if start_row < 0:
        raise ValueError("start_row must be >= 0")
    if max_rows < 0:
        raise ValueError("max_rows must be >= 0")

    df = pd.read_csv(csv_path)
    original_len = len(df)

    if start_row > 0:
        df = df.iloc[start_row:]
    if max_rows > 0:
        df = df.iloc[:max_rows]

    if len(df) == 0:
        raise ValueError("Selected subset is empty. Adjust start_row/max_rows.")

    timestamp_col = _find_timestamp_column(df)
    points_per_day = infer_points_per_day(df, timestamp_col)

    selected_target = choose_target_column(df, target_col)
    series = pd.to_numeric(df[selected_target], errors="coerce").dropna().astype(float)

    values = series.to_numpy(dtype=float)
    if len(values) < 30:
        raise ValueError("Series is too short. Need at least 30 points.")

    info = DatasetInfo(
        target_col_used=selected_target,
        timestamp_col_used=timestamp_col,
        inferred_points_per_day=points_per_day,
        num_rows=len(values),
        original_num_rows=original_len,
        subset_start_row=start_row,
        subset_max_rows=max_rows,
    )
    return values, info


def build_supervised_dataset(
    series: np.ndarray,
    mode: str,
    lag_count: int,
    points_per_day: int,
) -> Tuple[np.ndarray, np.ndarray, list]:
    if mode not in {"previous_values", "previous_days"}:
        raise ValueError("mode must be one of: previous_values, previous_days")

    if lag_count <= 0:
        raise ValueError("lag_count must be > 0")

    if mode == "previous_values":
        lags = list(range(1, lag_count + 1))
    else:
        if points_per_day <= 0:
            raise ValueError("points_per_day must be > 0 in previous_days mode")
        lags = [points_per_day * i for i in range(1, lag_count + 1)]

    max_lag = max(lags)
    if len(series) <= max_lag + 5:
        raise ValueError("Series too short for selected lag settings.")

    x_rows = []
    y_rows = []
    for t in range(max_lag, len(series)):
        x_rows.append([series[t - lag] for lag in lags])
        y_rows.append(series[t])

    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), lags


def time_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    validation_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.1 <= train_ratio <= 0.95):
        raise ValueError("train_ratio should be in [0.1, 0.95]")
    if not (0.0 <= validation_ratio <= 0.4):
        raise ValueError("validation_ratio should be in [0.0, 0.4]")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be < 1.0")

    n = len(x)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))

    train_end = min(max(train_end, 1), n - 2)
    val_end = min(max(val_end, train_end), n - 1)

    x_train = x[:train_end]
    y_train = y[:train_end]
    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]
    x_test = x[val_end:]
    y_test = y[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test
