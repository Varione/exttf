"""P1-2: Conservative NAV availability model shared by rotation signals.

Without official publication timestamps every signal must use a
pre-registered conservative availability lag: domestic funds are visible
one China trading day after the NAV date (T+1), QDII/cross-border funds two
trading days (T+2).  Signals built on NAV dated after the available cutoff
are forbidden and must be flagged.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

DOMESTIC_LAG_TRADING_DAYS = 1
QDII_LAG_TRADING_DAYS = 2

QDII_CODES: frozenset[str] = frozenset({"050025", "000071"})

MODEL_ID = "DOMESTIC_T1_QDII_T2"


def lag_for(code: str, qdii_codes: Iterable[str] = QDII_CODES) -> int:
    """Conservative availability lag in China trading days for ``code``."""
    normalized = str(code).strip().zfill(6)
    qdii = {str(c).strip().zfill(6) for c in qdii_codes}
    return QDII_LAG_TRADING_DAYS if normalized in qdii else DOMESTIC_LAG_TRADING_DAYS


def available_as_of(
    trading_dates: pd.DatetimeIndex,
    signal_date: pd.Timestamp,
    lag: int,
) -> pd.Timestamp:
    """Last China trading day whose NAV can be used at ``signal_date``.

    Falls back to the first trading date when the lag exceeds history.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values()
    position = int(dates.searchsorted(pd.Timestamp(signal_date), side="right") - 1)
    position = max(0, position - int(lag))
    return pd.Timestamp(dates[position])


def frame_as_of(
    frame: pd.DataFrame,
    code: str,
    available_date: pd.Timestamp,
    nav_column: str = "nav_date",
) -> pd.DataFrame:
    """Rows of ``frame`` for ``code`` with nav date at or before the cutoff."""
    return frame[
        (frame["fund_code"] == str(code).strip().zfill(6))
        & (pd.to_datetime(frame[nav_column]) <= pd.Timestamp(available_date))
    ].sort_values(nav_column)


def series_as_of(
    total_return_index: pd.DataFrame,
    code: str,
    available_date: pd.Timestamp,
) -> pd.Series:
    """Total-return index series for ``code`` at or before the cutoff."""
    if code not in total_return_index.columns:
        return pd.Series(dtype=float)
    return total_return_index.loc[: pd.Timestamp(available_date), code].dropna()


def availability_facts(
    qdii_codes: Iterable[str] = QDII_CODES,
) -> dict[str, object]:
    """Machine-readable description of the availability model."""
    return {
        "model_id": MODEL_ID,
        "domestic_lag_trading_days": DOMESTIC_LAG_TRADING_DAYS,
        "qdii_lag_trading_days": QDII_LAG_TRADING_DAYS,
        "qdii_codes": sorted(str(c).strip().zfill(6) for c in qdii_codes),
        "publication_timestamp_available": False,
        "conservative_availability_lag_applied": True,
        "label": "NAV_PUBLICATION_TIMESTAMP_NOT_AVAILABLE; CONSERVATIVE_AVAILABILITY_LAG_APPLIED",
    }
