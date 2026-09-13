from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "crypto-forecast-research/5.0 personal noncommercial research"}
BYBIT = "https://api.bybit.com"


def _rolling_z(s: pd.Series, n: int, minp: int | None = None) -> pd.Series:
    minp = minp or max(10, n // 2)
    m = s.rolling(n, min_periods=minp).mean()
    sd = s.rolling(n, min_periods=minp).std().replace(0, np.nan)
    return (s - m) / sd


def _request(path: str, params: dict, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(BYBIT + path, params=params, headers=UA, timeout=25)
            if r.ok:
                payload = r.json()
                if payload.get("retCode", 0) == 0:
                    return payload
                last = RuntimeError(f"Bybit retCode={payload.get('retCode')}: {payload.get('retMsg')}")
            else:
                last = RuntimeError(f"Bybit HTTP {r.status_code}")
        except Exception as e:
            last = e
        time.sleep(0.5 * (attempt + 1))
    if last:
        raise last
    return {}


def _windows(start: pd.Timestamp, end: pd.Timestamp, days: int = 150):
    cur = start.normalize()
    end = end.normalize()
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(days=1)


def _fetch_open_interest(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for a, b in _windows(start, end, 150):
        payload = _request("/v5/market/open-interest", {
            "category": "linear", "symbol": symbol, "intervalTime": "1d",
            "startTime": int(a.timestamp() * 1000),
            "endTime": int((b + pd.Timedelta(days=1)).timestamp() * 1000 - 1),
            "limit": 200,
        })
        rows.extend(payload.get("result", {}).get("list", []) or [])
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(pd.to_numeric(d["timestamp"], errors="coerce"), unit="ms", errors="coerce").dt.tz_localize(None).dt.normalize()
    d["oi"] = pd.to_numeric(d.get("openInterest"), errors="coerce")
    return d.dropna(subset=["date"]).groupby("date")[["oi"]].last().sort_index()


def _fetch_long_short(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Bybit documents history from 2020-07-20 onward.
    start = max(start, pd.Timestamp("2020-07-20"))
    rows = []
    for a, b in _windows(start, end, 300):
        payload = _request("/v5/market/account-ratio", {
            "category": "linear", "symbol": symbol, "period": "1d",
            "startTime": int(a.timestamp() * 1000),
            "endTime": int((b + pd.Timedelta(days=1)).timestamp() * 1000 - 1),
            "limit": 500,
        })
        rows.extend(payload.get("result", {}).get("list", []) or [])
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(pd.to_numeric(d["timestamp"], errors="coerce"), unit="ms", errors="coerce").dt.tz_localize(None).dt.normalize()
    d["long_share"] = pd.to_numeric(d.get("buyRatio"), errors="coerce")
    d["short_share"] = pd.to_numeric(d.get("sellRatio"), errors="coerce")
    d["ls_ratio"] = d["long_share"] / d["short_share"].replace(0, np.nan)
    return d.dropna(subset=["date"]).groupby("date")[["long_share", "short_share", "ls_ratio"]].last().sort_index()


def _fetch_premium(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for a, b in _windows(start, end, 900):
        payload = _request("/v5/market/premium-index-price-kline", {
            "category": "linear", "symbol": symbol, "interval": "D",
            "start": int(a.timestamp() * 1000),
            "end": int((b + pd.Timedelta(days=1)).timestamp() * 1000 - 1),
            "limit": 1000,
        })
        rows.extend(payload.get("result", {}).get("list", []) or [])
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    q = []
    for z in rows:
        if not isinstance(z, (list, tuple)) or len(z) < 5:
            continue
        q.append((z[0], z[1], z[2], z[3], z[4]))
    if not q:
        return pd.DataFrame()
    d = pd.DataFrame(q, columns=["timestamp", "premium_open", "premium_high", "premium_low", "premium_close"])
    d["date"] = pd.to_datetime(pd.to_numeric(d["timestamp"], errors="coerce"), unit="ms", errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ["premium_open", "premium_high", "premium_low", "premium_close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["premium_range"] = d["premium_high"] - d["premium_low"]
    return d.dropna(subset=["date"]).groupby("date")[["premium_close", "premium_range"]].last().sort_index()


def _engineer(d: pd.DataFrame, coin: str) -> pd.DataFrame:
    if d.empty:
        return d
    x = d.copy().sort_index()
    prefix = f"CUSTOM_BYBIT_{coin}_"

    if "oi" in x:
        oi = x["oi"].where(x["oi"] > 0)
        x[prefix + "oi_log"] = np.log(oi)
        x[prefix + "oi_chg1"] = oi.pct_change(1)
        x[prefix + "oi_chg7"] = oi.pct_change(7)
        x[prefix + "oi_chg30"] = oi.pct_change(30)
        x[prefix + "oi_z90"] = _rolling_z(np.log(oi), 90, 30)

    if "long_share" in x:
        x[prefix + "long_share"] = x["long_share"]
        x[prefix + "long_share_chg1"] = x["long_share"].diff(1)
        x[prefix + "long_share_chg7"] = x["long_share"].diff(7)
        x[prefix + "long_share_z90"] = _rolling_z(x["long_share"], 90, 30)
    if "ls_ratio" in x:
        x[prefix + "ls_ratio_log"] = np.log(x["ls_ratio"].where(x["ls_ratio"] > 0))
        x[prefix + "ls_ratio_z90"] = _rolling_z(x[prefix + "ls_ratio_log"], 90, 30)

    if "premium_close" in x:
        x[prefix + "premium"] = x["premium_close"]
        x[prefix + "premium_mean7"] = x["premium_close"].rolling(7, min_periods=3).mean()
        x[prefix + "premium_mean30"] = x["premium_close"].rolling(30, min_periods=10).mean()
        x[prefix + "premium_z90"] = _rolling_z(x["premium_close"], 90, 30)
    if "premium_range" in x:
        x[prefix + "premium_range"] = x["premium_range"]
        x[prefix + "premium_range_z90"] = _rolling_z(x["premium_range"], 90, 30)

    if prefix + "oi_chg7" in x and prefix + "premium_z90" in x:
        x[prefix + "leverage_pressure"] = x[prefix + "oi_chg7"] * x[prefix + "premium_z90"]
    if prefix + "long_share_z90" in x and prefix + "premium_z90" in x:
        x[prefix + "crowding"] = x[prefix + "long_share_z90"] * x[prefix + "premium_z90"]

    keep = [c for c in x.columns if c.startswith(prefix)]
    # One full-day lag is deliberate: only information safely known at the forecast timestamp is used.
    return x[keep].shift(1)


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d
    except Exception:
        return pd.DataFrame()


def _save_cache(path: Path, d: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = d.copy().sort_index()
    out.index.name = "date"
    out.to_csv(path)


def fetch_market_structure(index: pd.DatetimeIndex, cache_dir: Path, force: bool = False) -> Tuple[pd.DataFrame, Dict[str, object]]:
    end = min(pd.Timestamp.today().normalize() - pd.Timedelta(days=1), pd.Timestamp(index.max()).normalize())
    start = max(pd.Timestamp(index.min()).normalize(), pd.Timestamp("2020-07-20"))
    cache_path = cache_dir / "market_structure_v5.csv"
    old = _load_cache(cache_path)

    # Initial run builds the useful public history. Later runs only refresh the recent tail.
    if old.empty or force:
        fetch_start = start
    else:
        fetch_start = max(start, old.index.max() - pd.Timedelta(days=14))

    frames = []
    errors = []
    for coin, symbol in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        try:
            oi = _fetch_open_interest(symbol, fetch_start, end)
            ls = _fetch_long_short(symbol, fetch_start, end)
            prem = _fetch_premium(symbol, fetch_start, end)
            raw = pd.concat([oi, ls, prem], axis=1).sort_index()
            eng = _engineer(raw, coin)
            if not eng.empty:
                frames.append(eng)
        except Exception as e:
            errors.append(f"{coin}: {type(e).__name__}: {e}")

    fresh = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not old.empty:
        combined = pd.concat([old, fresh]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.loc[:, ~combined.columns.duplicated()]
    else:
        combined = fresh

    if not combined.empty:
        _save_cache(cache_path, combined)

    info = {
        "rows": int(len(combined)),
        "columns": int(len(combined.columns)) if not combined.empty else 0,
        "start": str(combined.index.min().date()) if not combined.empty else None,
        "end": str(combined.index.max().date()) if not combined.empty else None,
        "errors": errors,
    }
    return combined, info


def enrich_market_structure(df: pd.DataFrame, cache_dir: Path, force: bool = False) -> Tuple[pd.DataFrame, Dict[str, object]]:
    try:
        adv, info = fetch_market_structure(df.index, cache_dir, force=force)
        if adv.empty:
            return df, info
        # Prevent accidental duplicate columns if this function is called twice.
        add = [c for c in adv.columns if c not in df.columns]
        return df.join(adv[add].reindex(df.index), how="left"), info
    except Exception as e:
        return df, {"rows": 0, "columns": 0, "start": None, "end": None,
                    "errors": [f"{type(e).__name__}: {e}"]}
