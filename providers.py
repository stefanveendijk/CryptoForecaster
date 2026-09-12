from __future__ import annotations
import io
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import Config

UA = {"User-Agent": "crypto-forecast-research/4.0 personal noncommercial research"}

def _cache_csv(path: Path, loader, force=False) -> pd.DataFrame:
    if path.exists() and not force:
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df
    df = loader()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(path)
    return df

def fetch_yfinance_crypto(cfg: Config) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * cfg.history_years + 30))
    out = []
    for coin, ticker in cfg.coins.items():
        d = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), interval="1d",
                        auto_adjust=False, progress=False)
        if d.empty:
            raise RuntimeError(f"Geen prijsdata voor {ticker}")
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns]
        d = d[keep].rename(columns={c: f"{coin}_{c.lower().replace(' ','_')}" for c in keep})
        out.append(d)
    return pd.concat(out, axis=1).sort_index().ffill()

CROSS_ASSETS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "TLT": "TLT",
    "HYG": "HYG",
}

def fetch_cross_assets(cfg: Config) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * cfg.history_years + 60))
    frames = []
    for name, ticker in CROSS_ASSETS.items():
        try:
            d = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"), interval="1d",
                            auto_adjust=True, progress=False)
            if d.empty:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            s = pd.to_numeric(d["Close"], errors="coerce")
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            frames.append(s.rename(f"X_{name}_close"))
        except Exception:
            continue
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

FRED_SERIES = {
    "DFF": 1,          # effective fed funds
    "DGS10": 1,        # 10y treasury
    "DGS2": 1,         # 2y treasury
    "T10Y2": 1,        # curve
    "DTWEXBGS": 2,     # broad dollar index
    "STLFSI4": 7,      # weekly financial stress
    "WALCL": 7,        # Fed balance sheet, weekly
    "M2SL": 35,        # monthly/revised: conservative lag
    "CPIAUCSL": 35,    # monthly/revised: conservative lag
    "UNRATE": 14,      # monthly
}

def fetch_fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&cosd={start:%Y-%m-%d}&coed={end:%Y-%m-%d}"
    )
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    date_col = d.columns[0]
    value_col = d.columns[-1]
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    return d.dropna(subset=[date_col]).set_index(date_col)[value_col].rename(f"FRED_{series}")

def fetch_fred(cfg: Config) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(365.25 * cfg.history_years + 180))
    frames = []
    for series, lag in FRED_SERIES.items():
        try:
            s = fetch_fred_series(series, start, end).shift(lag, freq="D")
            frames.append(s)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, axis=1).sort_index()
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D")).ffill()

COINMETRICS_WANTED = [
    # Addresses / network activity
    "AdrActCnt", "AdrAct30dCnt", "AdrBalCnt", "AdrNewCnt", "AdrNewBalCnt",
    "TxCnt", "TxTfrCnt", "TxTfrValAdjUSD",
    # Valuation / holder state
    "CapMrktCurUSD", "CapMVRVCur", "CapRealUSD", "NVTAdj", "NVTAdj90",
    "RVT", "RVTAdj90", "SOPR", "NUPL",
    # Fees / economics
    "FeeTotUSD", "FeeMeanUSD",
    # Supply / issuance / coin activity
    "SplyCur", "SplyAct1d", "SplyAct7d", "SplyAct30d", "SplyAct1yr", "IssTotNtv",
    # PoW / security (will naturally be skipped for ETH post-merge or if unavailable)
    "HashRate", "DiffMean",
]

def _cm_catalog(asset: str) -> set:
    # Community catalog can vary by asset/metric; discover first.
    urls = [
        "https://community-api.coinmetrics.io/v4/catalog-all-v2/asset-metrics",
        "https://community-api.coinmetrics.io/v4/catalog/asset-metrics",
    ]
    for url in urls:
        try:
            r = requests.get(url, params={"assets": asset, "page_size": 10000},
                             headers=UA, timeout=60)
            if not r.ok:
                continue
            payload = r.json()
            metrics = set()
            for row in payload.get("data", []):
                if row.get("asset") == asset and isinstance(row.get("metrics"), list):
                    for m in row["metrics"]:
                        if isinstance(m, dict) and m.get("metric"):
                            metrics.add(m["metric"])
                if row.get("metric"):
                    metrics.add(row["metric"])
            if metrics:
                return metrics
        except Exception:
            continue
    return set()

def fetch_coinmetrics_asset(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    available = _cm_catalog(asset)
    wanted = [m for m in COINMETRICS_WANTED if not available or m in available]
    if not wanted:
        return pd.DataFrame()

    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": asset,
        "metrics": ",".join(wanted),
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
        "page_size": 10000,
        "paging_from": "start",
        "ignore_forbidden_errors": "true",
        "ignore_unsupported_errors": "true",
    }
    rows = []
    token = None
    while True:
        if token:
            params["next_page_token"] = token
        r = requests.get(url, params=params, headers=UA, timeout=90)
        if not r.ok:
            # Retry metric-by-metric if community permissions differ.
            parts = []
            for metric in wanted:
                p = params.copy()
                p["metrics"] = metric
                p.pop("next_page_token", None)
                rr = requests.get(url, params=p, headers=UA, timeout=60)
                if rr.ok:
                    dat = rr.json().get("data", [])
                    if dat:
                        q = pd.DataFrame(dat)
                        if "time" in q:
                            q["date"] = pd.to_datetime(q["time"], errors="coerce").dt.tz_localize(None).dt.normalize()
                            q[metric] = pd.to_numeric(q.get(metric), errors="coerce")
                            parts.append(q.set_index("date")[[metric]])
                time.sleep(0.2)
            return pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()
        payload = r.json()
        rows.extend(payload.get("data", []))
        token = payload.get("next_page_token")
        if not token:
            break
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["time"], errors="coerce").dt.tz_localize(None).dt.normalize()
    cols = []
    for m in wanted:
        if m in d:
            d[m] = pd.to_numeric(d[m], errors="coerce")
            cols.append(m)
    return d.set_index("date")[cols].sort_index()

def fetch_coinmetrics(cfg: Config) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * cfg.history_years + 90))
    frames = []
    for coin, asset in [("BTC","btc"), ("ETH","eth")]:
        try:
            d = fetch_coinmetrics_asset(asset, start, end)
            if not d.empty:
                frames.append(d.add_prefix(f"OC_{coin}_"))
        except Exception:
            continue

    # Stablecoin supply/liquidity proxy. Only use metrics that community exposes.
    for stable in ["usdt", "usdc"]:
        try:
            d = fetch_coinmetrics_asset(stable, start, end)
            keep = [c for c in ["SplyCur","CapMrktCurUSD"] if c in d.columns]
            if keep:
                frames.append(d[keep].add_prefix(f"STABLE_{stable.upper()}_"))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().shift(cfg.onchain_lag_days)

def fetch_fear_greed(cfg: Config) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    r = requests.get(url, params={"limit": 0, "format": "json"}, headers=UA, timeout=60)
    r.raise_for_status()
    rows = r.json().get("data", [])
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["date"] = pd.to_datetime(pd.to_numeric(d["timestamp"]), unit="s").dt.tz_localize(None).dt.normalize()
    d["FGI_value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.set_index("date")[["FGI_value"]].sort_index().shift(cfg.exogenous_lag_days)

def _binance_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    start_ms = int(start.timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=1)).timestamp() * 1000)
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(url, params={
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000
        }, headers=UA, timeout=60)
        if not r.ok:
            break
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1]["fundingTime"]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(pd.to_numeric(d["fundingTime"]), unit="ms").dt.tz_localize(None).dt.normalize()
    d["fundingRate"] = pd.to_numeric(d["fundingRate"], errors="coerce")
    return d.groupby("date")["fundingRate"].agg(["mean","sum","std","max","min"])

def fetch_binance_derivatives(cfg: Config) -> pd.DataFrame:
    # Funding history is suitable for historical training; public OI history is only ~30 days,
    # so OI is intentionally not mixed into the long backtest.
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * cfg.history_years))
    frames = []
    for coin, symbol in [("BTC","BTCUSDT"), ("ETH","ETHUSDT")]:
        try:
            d = _binance_funding(symbol, start, end)
            if not d.empty:
                d = d.add_prefix(f"DERIV_{coin}_funding_")
                frames.append(d)
        except Exception:
            continue
    return pd.concat(frames, axis=1).sort_index().shift(cfg.exogenous_lag_days) if frames else pd.DataFrame()

def _parse_parentheses_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in {"-", "", "nan"}:
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan

def _farside_table(url: str, prefix: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    candidates = []
    for t in tables:
        if t.shape[1] >= 3:
            first = t.iloc[:,0].astype(str)
            if first.str.contains(r"\d{2}\s+[A-Za-z]{3}\s+\d{4}", regex=True).any():
                candidates.append(t)
    if not candidates:
        return pd.DataFrame()
    t = max(candidates, key=len).copy()
    date_raw = t.iloc[:,0]
    total = t.iloc[:,-1]
    d = pd.DataFrame({
        "date": pd.to_datetime(date_raw, dayfirst=True, errors="coerce"),
        f"{prefix}_netflow_usdm": total.map(_parse_parentheses_number)
    }).dropna(subset=["date"])
    return d.set_index("date").sort_index()

def fetch_etf_flows(cfg: Config) -> pd.DataFrame:
    frames = []
    for prefix, url in [
        ("ETF_BTC", "https://farside.co.uk/bitcoin-etf-flow-all-data/"),
        ("ETF_ETH", "https://farside.co.uk/ethereum-etf-flow-all-data/"),
    ]:
        try:
            d = _farside_table(url, prefix)
            if not d.empty:
                frames.append(d)
        except Exception:
            continue
    return pd.concat(frames, axis=1).sort_index().shift(cfg.etf_lag_days) if frames else pd.DataFrame()


def _deribit_dvol(currency: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Deribit volatility-index candles (DVOL-like implied volatility).
    Public endpoint; history is naturally shorter than the 10-year price sample.
    """
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    start_ms = int(start.timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=1)).timestamp() * 1000)
    rows = []
    cursor_end = end_ms
    seen = set()

    for _ in range(200):
        params = {
            "currency": currency,
            "start_timestamp": start_ms,
            "end_timestamp": cursor_end,
            "resolution": "1D",
        }
        r = requests.get(url, params=params, headers=UA, timeout=60)
        if not r.ok:
            break
        result = r.json().get("result", {})
        batch = result.get("data", []) if isinstance(result, dict) else []
        for z in batch:
            if len(z) >= 5 and z[0] not in seen:
                seen.add(z[0]); rows.append(z[:5])
        cont = result.get("continuation") if isinstance(result, dict) else None
        if cont is None or not batch:
            break
        try:
            cont = int(cont)
        except Exception:
            break
        if cont >= cursor_end:
            break
        cursor_end = cont
        time.sleep(0.10)

    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows, columns=["timestamp","open","high","low","close"])
    d["date"] = pd.to_datetime(pd.to_numeric(d["timestamp"]), unit="ms").dt.tz_localize(None).dt.normalize()
    for c in ["open","high","low","close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.set_index("date")[["open","high","low","close"]].sort_index()

def fetch_deribit_volatility(cfg: Config) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * cfg.history_years))
    frames = []
    for coin in ["BTC","ETH"]:
        try:
            d = _deribit_dvol(coin, start, end)
            if not d.empty:
                frames.append(d.add_prefix(f"OPTIONS_{coin}_dvol_"))
        except Exception:
            continue
    return pd.concat(frames, axis=1).sort_index().shift(cfg.exogenous_lag_days) if frames else pd.DataFrame()


def fetch_google_trends(cfg: Config) -> pd.DataFrame:
    # Optional because pytrends is unofficial and may break if Google changes its frontend.
    try:
        from pytrends.request import TrendReq
    except Exception:
        return pd.DataFrame()
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(365.25 * cfg.history_years))
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(["bitcoin", "ethereum"], timeframe=f"{start:%Y-%m-%d} {end:%Y-%m-%d}")
    d = pytrends.interest_over_time()
    if d.empty:
        return d
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    cols = {}
    if "bitcoin" in d: cols["bitcoin"] = "TRENDS_bitcoin"
    if "ethereum" in d: cols["ethereum"] = "TRENDS_ethereum"
    d = d.rename(columns=cols)[list(cols.values())]
    daily = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D")).ffill()
    return daily.shift(cfg.trends_lag_days)

def load_custom_features(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        frames.append(d.add_prefix(f"CUSTOM_{path.stem}_").shift(1))
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

def gather_all_nonnews(cfg: Config, cache_dir: Path, force=False) -> Dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    out["crypto"] = fetch_yfinance_crypto(cfg)

    loaders = []
    if cfg.enable_cross_assets:
        loaders.append(("cross", fetch_cross_assets))
    if cfg.enable_fred:
        loaders.append(("fred", fetch_fred))
    if cfg.enable_coinmetrics:
        loaders.append(("onchain", fetch_coinmetrics))
    if cfg.enable_fear_greed:
        loaders.append(("sentiment", fetch_fear_greed))
    if cfg.enable_binance_derivatives:
        loaders.append(("derivatives", fetch_binance_derivatives))
        loaders.append(("options", fetch_deribit_volatility))
    if cfg.enable_etf_flows:
        loaders.append(("etf", fetch_etf_flows))
    if cfg.enable_google_trends:
        loaders.append(("trends", fetch_google_trends))

    for name, loader in loaders:
        path = cache_dir / f"{name}.csv"
        try:
            out[name] = _cache_csv(path, lambda l=loader: l(cfg), force=force)
        except Exception as e:
            print(f"WAARSCHUWING bron {name}: {e}")
            out[name] = pd.DataFrame()

    out["custom"] = load_custom_features(cfg.custom_feature_files)
    return out
