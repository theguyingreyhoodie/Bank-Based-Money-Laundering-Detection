"""Feature engineering for transaction-level AML detection.

Every account-level feature uses only transactions that happened BEFORE the current one
(for the same account), so no information leaks from the future. The label is never used.

Feature pool (column names):
    Transaction : log_amount, payment_format, payment_currency, receiving_currency,
                  cross_currency, cross_bank, is_round, hour, dayofweek
    Sender      : src_txn_count, src_amount_mean, src_amount_std, src_fan_out,
                  src_num_currencies, src_secs_since_prev, src_burst
    Receiver    : dst_fan_in, dst_received_volume, dst_burst

Who can compute what: all of a sender's payments go through the sender's own bank, so a bank
sees the whole history behind the sender features. A receiver, though, is paid by customers of
many banks, and each bank sees only the payments its own customers make. build_features counts
every bank's payments (the pooled, centralized view); receiver_features(..., scope=...) counts
only one bank's, which is what prepare_data.py gives each bank's partition.

Caveat: the raw timestamps have minute resolution. Transactions in the same minute are
ordered by file order, so "previous" means "earlier in the file" within a minute.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency"]
NUMERIC = [
    "log_amount", "cross_currency", "cross_bank", "is_round", "hour", "dayofweek",
    "src_txn_count", "src_amount_mean", "src_amount_std", "src_fan_out",
    "src_num_currencies", "src_secs_since_prev", "src_burst",
    "dst_fan_in", "dst_received_volume", "dst_burst",
]
TXN_FEATURES = CATEGORICAL + NUMERIC
RECEIVER_FEATURES = ["dst_fan_in", "dst_received_volume", "dst_burst"]

# Public category lists (payment formats and currencies of the payment network), agreed by
# all banks in advance so that one-hot columns line up when a whole model is averaged.
# Values follow the IBM AML data; anything else is encoded as "<other>". Check them against
# "payment_formats" / "currencies" in data/processed/summary.json for the real data.
PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = ["Australian Dollar", "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro",
              "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal", "Shekel", "Swiss Franc",
              "UK Pound", "US Dollar", "Yen", "Yuan"]
SHARED_VOCAB = {"payment_format": PAYMENT_FORMATS, "payment_currency": CURRENCIES,
                "receiving_currency": CURRENCIES}

# Value used for "no previous transaction" in time-gap features (30 days, in seconds).
NO_PREV_SECS = 30 * 24 * 3600.0


def _prior_distinct_count(key: pd.Series, other: pd.Series) -> pd.Series:
    """For each row: how many distinct `other` values the row's `key` had in earlier rows."""
    is_new = ~pd.DataFrame({"key": key, "other": other}).duplicated()
    return is_new.groupby(key).cumsum() - is_new.astype(int)


def _prior_window_count(key: pd.Series, timestamp: pd.Series, hours: int) -> pd.Series:
    """For each row: number of earlier rows for the same `key` within the last `hours`."""
    codes = pd.factorize(key)[0].astype(np.int64)
    minutes = ((timestamp - timestamp.min()).dt.total_seconds() // 60).astype(np.int64).to_numpy()
    big = int(minutes.max()) + hours * 60 + 1
    # Rows are time-sorted, so a stable sort by key keeps time order inside each key.
    order = np.argsort(codes, kind="stable")
    composite = codes[order] * big + minutes[order]          # sorted ascending
    pos = np.arange(len(order))
    # first earlier row of the same key that is still inside (t - window, t]
    lower = np.searchsorted(composite, composite - hours * 60, side="right")
    out = np.empty(len(order), dtype=np.float32)
    out[order] = pos - lower
    return pd.Series(out, index=key.index)


def receiver_features(df: pd.DataFrame, burst_window_hours: int = 24,
                      scope: pd.Series | None = None) -> pd.DataFrame:
    """dst_fan_in, dst_received_volume and dst_burst: the receiver's history before each payment.

    scope=None counts every earlier payment to the receiver, which takes every bank's data (the
    pooled view). With `scope`, a Series aligned with df such as the client bank of each row, a
    row counts only earlier payments with the same scope value: what that bank sees in its own
    records. Every sender belongs to exactly one bank, so each pooled value is the sum of the
    per-bank values (a secure sum across banks could therefore recover it).
    """
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Input must be sorted by timestamp (use data.clean first).")
    key = df["dst_key"]
    if scope is not None:
        if scope.isna().any():
            raise ValueError("scope has missing values")
        # One integer per (scope, receiver) pair.
        k, s = pd.factorize(key)[0].astype(np.int64), pd.factorize(scope)[0].astype(np.int64)
        key = pd.Series(k * (int(s.max()) + 1) + s, index=df.index)
    amt = df["amount_usd"]
    received = amt.groupby(key).cumsum() - amt
    return pd.DataFrame({
        "dst_fan_in": _prior_distinct_count(key, df["src_key"]).astype(np.float32),
        "dst_received_volume": np.log1p(received.clip(lower=0)).astype(np.float32),
        "dst_burst": _prior_window_count(key, df["timestamp"], burst_window_hours),
    }, index=df.index)


def build_features(df: pd.DataFrame, burst_window_hours: int = 24,
                   round_amount_min: float = 100.0) -> pd.DataFrame:
    """Add all pool features to a cleaned, time-sorted frame (see data.clean)."""
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Input must be sorted by timestamp (use data.clean first).")
    out = df.copy()
    amt = out["amount_usd"]

    # --- transaction-level -------------------------------------------------------
    out["log_amount"] = np.log1p(amt).astype(np.float32)
    out["cross_currency"] = (out["payment_currency"] != out["receiving_currency"]).astype(np.int8)
    out["cross_bank"] = (out["src_bank"] != out["dst_bank"]).astype(np.int8)
    out["is_round"] = ((out["amount_paid"] >= round_amount_min)
                       & (np.isclose(out["amount_paid"] % 100, 0))).astype(np.int8)
    out["hour"] = out["timestamp"].dt.hour.astype(np.int8)
    out["dayofweek"] = out["timestamp"].dt.dayofweek.astype(np.int8)

    # --- sender history (past only) ---------------------------------------------
    g = out.groupby("src_key")
    out["src_txn_count"] = g.cumcount().astype(np.float32)
    prev_sum = g["log_amount"].cumsum() - out["log_amount"]
    prev_sq = (out["log_amount"] ** 2).groupby(out["src_key"]).cumsum() - out["log_amount"] ** 2
    n = out["src_txn_count"].replace(0, np.nan)
    mean = prev_sum / n
    var = (prev_sq / n - mean ** 2).clip(lower=0)
    out["src_amount_mean"] = mean.fillna(0).astype(np.float32)
    out["src_amount_std"] = np.sqrt(var).fillna(0).astype(np.float32)
    out["src_fan_out"] = _prior_distinct_count(out["src_key"], out["dst_key"]).astype(np.float32)
    out["src_num_currencies"] = _prior_distinct_count(out["src_key"], out["payment_currency"]).astype(np.float32)
    gap = g["timestamp"].diff().dt.total_seconds()
    out["src_secs_since_prev"] = gap.fillna(NO_PREV_SECS).astype(np.float32)
    out["src_burst"] = _prior_window_count(out["src_key"], out["timestamp"], burst_window_hours)

    # --- receiver history (past only; every bank's payments, see receiver_features) --
    for col, values in receiver_features(out, burst_window_hours).items():
        out[col] = values

    log.info("Built %d features for %d transactions", len(TXN_FEATURES), len(out))
    return out
