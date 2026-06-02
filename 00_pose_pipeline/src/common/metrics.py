"""Small metric helpers used by angle and motion evaluation."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired finite values."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Pearson correlation with guardrails."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) < 3 or np.nanstd(x_f) < 1e-9 or np.nanstd(y_f) < 1e-9:
        return None
    return float(np.corrcoef(x_f, y_f)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Spearman correlation with guardrails."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) < 3 or np.nanstd(x_f) < 1e-9 or np.nanstd(y_f) < 1e-9:
        return None
    result = spearmanr(x_f, y_f)
    value = result.correlation if hasattr(result, "correlation") else result[0]
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def mae(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Mean absolute difference for finite pairs."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) == 0:
        return None
    return float(np.mean(np.abs(x_f - y_f)))


def median_abs_error(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Median absolute difference for finite pairs."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) == 0:
        return None
    return float(np.median(np.abs(x_f - y_f)))


def rmse(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Root mean squared difference for finite pairs."""
    x_f, y_f = finite_pair(x, y)
    if len(x_f) == 0:
        return None
    return float(np.sqrt(np.mean((x_f - y_f) ** 2)))


def regression_slope(reference: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Least-squares target = slope * reference + intercept slope."""
    ref, tgt = finite_pair(reference, target)
    if len(ref) < 3 or np.nanvar(ref) < 1e-12:
        return None
    return float(np.cov(ref, tgt, bias=True)[0, 1] / np.var(ref))


def rula_bin(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Assign simple RULA-like bins from angle thresholds."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    out[finite] = np.searchsorted(np.asarray(thresholds, dtype=np.float64), values[finite], side="right")
    return out


def jsonable(value):
    """Convert NumPy values to JSON-friendly objects."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value
