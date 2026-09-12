"""
BTC/ETH Predictor v2 — koers + nieuws
=====================================

Onderzoeksdoel
--------------
1. Download circa 10 jaar dagelijkse BTC- en ETH-koersen.
2. Download historische nieuwsintensiteit en nieuwstoon via GDELT DOC 2.0.
3. Onderzoek lead/lag-relaties tussen nieuws en rendement.
4. Vergelijk eerlijk, out-of-sample:
      A) prijsmodel
      B) prijs + nieuwsmodel
5. Geef alleen een "nieuws-edge" wanneer B de benchmark A daadwerkelijk verbetert.

Belangrijk
----------
- GDELT DOC 2.0 is publiek doorzoekbaar vanaf 1 januari 2017. Daardoor is de
  gratis nieuwscomponent ongeveer 9,5+ jaar in plaats van exact 10 jaar.
  De prijscomponent gebruikt wel circa 10 jaar.
- Alle nieuwsfeatures worden standaard 1 dag vertraagd, zodat geen nieuws uit
  de toekomst of uit een onvolledig handels-/kalenderdag wordt gebruikt.
- Correlatie is geen causaliteit. Daarom bevat het programma ook Granger-tests,
  reverse-direction tests en walk-forward modelvergelijking.
- Dit is een onderzoeksinstrument, geen beleggingsadvies of rendementsbelofte.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests


COINS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
HORIZONS = [7, 30]
START_YEARS_AGO = 10
MIN_TRAIN_YEARS = 4
RETRAIN_EVERY_DAYS = 30
TRADING_COST = 0.001

# GDELT DOC 2.0 historische online-nieuwszoekindex.
GDELT_START = pd.Timestamp("2017-01-01")
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_FEATURE_LAG_DAYS = 1

# Hoofdzoektermen. We vermijden "ether" vanwege veel niet-crypto false positives.
COIN_QUERY = {
    "BTC": 'bitcoin',
    "ETH": 'ethereum',
}

# Niet geneste OR-blokken; GDELT interpreteert losse termen/blokken als aanvullende
# zoekvoorwaarden. Deze categorieën meten volume, niet direct "waarheid" van het nieuws.
TOPIC_QUERIES = {
    "regulation": '(regulation OR regulator OR regulated OR SEC OR lawsuit OR ban OR law)',
    "security": '(hack OR hacked OR exploit OR theft OR scam OR fraud OR breach)',
    "etf_institutional": '(ETF OR institutional OR institution OR BlackRock OR Fidelity OR adoption)',
}


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> Tuple[float, float, int]:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 50 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, len(z)
    if method == "pearson":
        r, p = pearsonr(z.iloc[:, 0], z.iloc[:, 1])
    else:
        r, p = spearmanr(z.iloc[:, 0], z.iloc[:, 1])
    return float(r), float(p), len(z)


def download_price_data() -> pd.DataFrame:
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=int(365.25 * START_YEARS_AGO))

    frames = {}
    for name, ticker in COINS.items():
        d = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if d.empty:
            raise RuntimeError(f"Geen koersdata ontvangen voor {ticker}.")
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
        d = d[keep].copy()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        frames[name] = d

    idx = frames["BTC"].index.union(frames["ETH"].index).sort_values()
    out = pd.DataFrame(index=idx)
    for name, d in frames.items():
        for c in d.columns:
            out[f"{name}_{c.lower()}"] = d[c].reindex(idx)

    return out.sort_index().ffill()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    for coin in COINS:
        p = x[f"{coin}_close"]
        v = x[f"{coin}_volume"].replace(0, np.nan)

        x[f"{coin}_ret1"] = np.log(p).diff()

        for n in [2, 3, 7, 14, 30, 60, 90]:
            x[f"{coin}_mom_{n}"] = p.pct_change(n)

        for n in [7, 14, 30, 60, 90]:
            x[f"{coin}_vol_{n}"] = x[f"{coin}_ret1"].rolling(n).std() * np.sqrt(365)

        for n in [7, 14, 30, 90, 200]:
            ma = p.rolling(n).mean()
            x[f"{coin}_ma_ratio_{n}"] = p / ma - 1

        x[f"{coin}_rsi14"] = rsi(p, 14) / 100.0
        x[f"{coin}_range"] = (x[f"{coin}_high"] - x[f"{coin}_low"]) / p
        x[f"{coin}_oc"] = (x[f"{coin}_close"] - x[f"{coin}_open"]) / x[f"{coin}_open"]

        lv = np.log(v)
        x[f"{coin}_volume_z30"] = (lv - lv.rolling(30).mean()) / lv.rolling(30).std()
        x[f"{coin}_volume_chg7"] = lv.diff(7)

        rollmax30 = p.rolling(30).max()
        rollmax90 = p.rolling(90).max()
        x[f"{coin}_drawdown30"] = p / rollmax30 - 1
        x[f"{coin}_drawdown90"] = p / rollmax90 - 1

        for h in HORIZONS:
            x[f"{coin}_target_ret_{h}"] = p.shift(-h) / p - 1
            x[f"{coin}_target_up_{h}"] = (x[f"{coin}_target_ret_{h}"] > 0).astype(float)
            x.loc[x[f"{coin}_target_ret_{h}"].isna(), f"{coin}_target_up_{h}"] = np.nan

    x["cross_ret_spread_1"] = x["BTC_ret1"] - x["ETH_ret1"]
    x["cross_mom_spread_30"] = x["BTC_mom_30"] - x["ETH_mom_30"]
    x["cross_corr_30"] = x["BTC_ret1"].rolling(30).corr(x["ETH_ret1"])
    x["cross_corr_90"] = x["BTC_ret1"].rolling(90).corr(x["ETH_ret1"])
    x["eth_btc_ratio"] = x["ETH_close"] / x["BTC_close"]
    x["eth_btc_mom30"] = x["eth_btc_ratio"].pct_change(30)

    return x.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# GDELT nieuws
# ---------------------------------------------------------------------------

def _gdelt_request(query: str, mode: str, start: pd.Timestamp, end: pd.Timestamp,
                   timeout: int = 90) -> dict:
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "STARTDATETIME": start.strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME": end.strftime("%Y%m%d%H%M%S"),
    }
    headers = {
        "User-Agent": "BTC-ETH-research-predictor/2.0 (personal research; low-frequency requests)"
    }
    r = requests.get(GDELT_ENDPOINT, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()

    ctype = r.headers.get("content-type", "")
    if "json" not in ctype.lower() and not r.text.lstrip().startswith("{"):
        raise RuntimeError(f"GDELT gaf geen JSON terug: {r.text[:180]}")

    return r.json()


def _parse_gdelt_timeline(payload: dict, mode: str) -> pd.DataFrame:
    timeline = payload.get("timeline", [])
    if not timeline:
        return pd.DataFrame()

    first_data = timeline[0].get("data", [])
    if not first_data:
        return pd.DataFrame()

    out = pd.DataFrame({
        "date": pd.to_datetime([z.get("date") for z in first_data], errors="coerce")
    })
    out["date"] = out["date"].dt.tz_localize(None).dt.normalize()

    for series in timeline:
        name = str(series.get("series", "series")).strip() or "series"
        vals = [z.get("value", np.nan) for z in series.get("data", [])]
        if len(vals) == len(out):
            out[name] = pd.to_numeric(vals, errors="coerce")

    if mode == "timelinevolraw":
        norm = [z.get("norm", np.nan) for z in first_data]
        if len(norm) == len(out):
            out["All Articles"] = pd.to_numeric(norm, errors="coerce")

    return out.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()


def fetch_gdelt_timeline(
    query: str,
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = 180,
) -> pd.DataFrame:
    """
    Probeert eerst één historische query. Als de API die range niet accepteert,
    valt de functie terug op kleinere blokken.
    """
    try:
        payload = _gdelt_request(query, mode, start, end)
        frame = _parse_gdelt_timeline(payload, mode)
        if not frame.empty:
            return frame
    except Exception:
        pass

    pieces = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end)
        last_error = None
        for attempt in range(3):
            try:
                payload = _gdelt_request(query, mode, cur, nxt)
                frame = _parse_gdelt_timeline(payload, mode)
                if not frame.empty:
                    pieces.append(frame)
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            print(f"WAARSCHUWING GDELT {cur.date()}–{nxt.date()}: {last_error}")
        cur = nxt
        time.sleep(0.20)

    if not pieces:
        return pd.DataFrame()

    return (
        pd.concat(pieces)
        .sort_index()
        .groupby(level=0)
        .last()
    )


def _series_from_timeline(frame: pd.DataFrame, prefer_names: Iterable[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    lower = {c.lower(): c for c in frame.columns}
    for p in prefer_names:
        for lc, orig in lower.items():
            if p.lower() in lc:
                return pd.to_numeric(frame[orig], errors="coerce")
    numeric = [c for c in frame.columns if c != "All Articles"]
    if numeric:
        return pd.to_numeric(frame[numeric[0]], errors="coerce")
    return pd.Series(index=frame.index, dtype=float)


def download_news_for_coin(coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    base_query = COIN_QUERY[coin]
    print(f"  GDELT nieuws ophalen voor {coin}: {start.date()} t/m {end.date()}")

    vol = fetch_gdelt_timeline(base_query, "timelinevolraw", start, end)
    tone = fetch_gdelt_timeline(base_query, "timelinetone", start, end)

    neg = fetch_gdelt_timeline(f'{base_query} tone<-5', "timelinevolraw", start, end)
    pos = fetch_gdelt_timeline(f'{base_query} tone>5', "timelinevolraw", start, end)

    idx = pd.date_range(start.normalize(), end.normalize(), freq="D")
    out = pd.DataFrame(index=idx)

    out["article_count"] = _series_from_timeline(vol, [base_query, "volume"]).reindex(idx)
    if "All Articles" in vol:
        out["all_articles"] = pd.to_numeric(vol["All Articles"], errors="coerce").reindex(idx)
    else:
        out["all_articles"] = np.nan

    out["tone"] = _series_from_timeline(tone, ["tone", base_query]).reindex(idx)
    out["negative_count"] = _series_from_timeline(neg, [base_query, "volume"]).reindex(idx)
    out["positive_count"] = _series_from_timeline(pos, [base_query, "volume"]).reindex(idx)

    for topic, q in TOPIC_QUERIES.items():
        tf = fetch_gdelt_timeline(f"{base_query} {q}", "timelinevolraw", start, end)
        out[f"{topic}_count"] = _series_from_timeline(tf, [base_query, "volume"]).reindex(idx)

    # Een ontbrekende volume-observatie behandelen we alleen als 0 wanneer GDELT
    # voor die dag wél een totale coverage ("all_articles") heeft.
    count_cols = [c for c in out.columns if c.endswith("_count") or c == "article_count"]
    for c in count_cols:
        out.loc[out["all_articles"].notna() & out[c].isna(), c] = 0.0

    # Geen artikelen -> neutrale toon 0. Anders ontbrekende toon als NaN laten.
    out.loc[(out["article_count"] == 0) & out["tone"].isna(), "tone"] = 0.0

    return out


def load_or_update_news(
    cache_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_coins = []

    start = max(start.normalize(), GDELT_START)
    end = end.normalize()

    for coin in COINS:
        path = cache_dir / f"gdelt_{coin.lower()}_daily.csv"

        existing = pd.DataFrame()
        if path.exists() and not force:
            existing = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()

        ranges = []
        if force or existing.empty:
            ranges = [(start, end)]
        else:
            existing.index = pd.to_datetime(existing.index).tz_localize(None).normalize()
            have_start, have_end = existing.index.min(), existing.index.max()
            if have_start > start:
                ranges.append((start, have_start - pd.Timedelta(days=1)))
            if have_end < end:
                ranges.append((have_end + pd.Timedelta(days=1), end))

        pieces = [existing] if not existing.empty else []
        for a, b in ranges:
            if a <= b:
                pieces.append(download_news_for_coin(coin, a, b))

        if not pieces:
            raise RuntimeError(f"Geen nieuwsdata voor {coin}.")

        coin_df = pd.concat(pieces).sort_index()
        coin_df = coin_df[~coin_df.index.duplicated(keep="last")]
        coin_df.index.name = "date"
        coin_df.to_csv(path)

        coin_df = coin_df.add_prefix(f"{coin}_news_")
        all_coins.append(coin_df)

    return pd.concat(all_coins, axis=1).sort_index()


def add_news_features(df: pd.DataFrame, raw_news: pd.DataFrame) -> pd.DataFrame:
    x = df.join(raw_news, how="left")

    for coin in COINS:
        pfx = f"{coin}_news_"
        count = x[f"{pfx}article_count"]
        total = x[f"{pfx}all_articles"].replace(0, np.nan)
        tone = x[f"{pfx}tone"]

        x[f"{pfx}volume_share"] = count / total
        x[f"{pfx}negative_share"] = x[f"{pfx}negative_count"] / count.replace(0, np.nan)
        x[f"{pfx}positive_share"] = x[f"{pfx}positive_count"] / count.replace(0, np.nan)
        x[f"{pfx}sentiment_balance"] = (
            x[f"{pfx}positive_count"] - x[f"{pfx}negative_count"]
        ) / (count + 1.0)

        for topic in TOPIC_QUERIES:
            x[f"{pfx}{topic}_share"] = x[f"{pfx}{topic}_count"] / (count + 1.0)

        lv = np.log1p(count)

        for n in [3, 7, 30]:
            x[f"{pfx}tone_ma{n}"] = tone.rolling(n).mean()
            x[f"{pfx}volume_ma{n}"] = x[f"{pfx}volume_share"].rolling(n).mean()
            x[f"{pfx}neg_ma{n}"] = x[f"{pfx}negative_share"].rolling(n).mean()
            x[f"{pfx}pos_ma{n}"] = x[f"{pfx}positive_share"].rolling(n).mean()

        for n in [30, 90]:
            x[f"{pfx}volume_z{n}"] = (lv - lv.rolling(n).mean()) / lv.rolling(n).std()
            x[f"{pfx}tone_z{n}"] = (tone - tone.rolling(n).mean()) / tone.rolling(n).std()
            for topic in TOPIC_QUERIES:
                s = np.log1p(x[f"{pfx}{topic}_count"])
                x[f"{pfx}{topic}_z{n}"] = (s - s.rolling(n).mean()) / s.rolling(n).std()

        for lag in [1, 3, 7]:
            x[f"{pfx}tone_change{lag}"] = tone.diff(lag)
            x[f"{pfx}volume_change{lag}"] = lv.diff(lag)

    # Anti-leakage:
    # nieuws van kalenderdag t wordt pas vanaf t+1 als voorspellende feature gebruikt.
    news_cols = [c for c in x.columns if "_news_" in c]
    x[news_cols] = x[news_cols].shift(NEWS_FEATURE_LAG_DAYS)

    return x.replace([np.inf, -np.inf], np.nan)


def base_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if "target_" in c or "_news_" in c:
            continue
        if c.endswith(("_open", "_high", "_low", "_close", "_volume")):
            continue
        cols.append(c)
    return cols


def news_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Alleen afgeleide nieuwsfeatures gebruiken; de ruwe globale All Articles wordt
    uitgesloten. Ruwe counts mogen wel mee naast genormaliseerde intensiteit.
    """
    cols = []
    for c in df.columns:
        if "_news_" not in c:
            continue
        if c.endswith("all_articles"):
            continue
        cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Relaties / correlaties / causaliteitsrichting
# ---------------------------------------------------------------------------

def correlation_study(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for coin in COINS:
        ret1 = df[f"{coin}_ret1"]
        pfx = f"{coin}_news_"

        candidates = {
            "tone": df.get(f"{pfx}tone"),
            "volume_share": df.get(f"{pfx}volume_share"),
            "negative_share": df.get(f"{pfx}negative_share"),
            "positive_share": df.get(f"{pfx}positive_share"),
            "security_share": df.get(f"{pfx}security_share"),
            "regulation_share": df.get(f"{pfx}regulation_share"),
            "etf_institutional_share": df.get(f"{pfx}etf_institutional_share"),
        }

        for feature_name, s in candidates.items():
            if s is None:
                continue

            # De stored newsfeature is al 1 dag vertraagd.
            # "future_h" = return vanaf huidige close tot h dagen later.
            for h in [1, 3, 7, 30]:
                future_ret = df[f"{coin}_close"].shift(-h) / df[f"{coin}_close"] - 1
                for method in ["pearson", "spearman"]:
                    r, p, n = _safe_corr(s, future_ret, method)
                    rows.append({
                        "coin": coin,
                        "direction": "news_to_future_return",
                        "feature": feature_name,
                        "horizon_days": h,
                        "method": method,
                        "correlation": r,
                        "p_value": p,
                        "n": n,
                    })

            # Reverse-direction: kan koersbeweging nieuws/sentiment voorspellen?
            for lead in [1, 3, 7]:
                future_news = s.shift(-lead)
                for method in ["pearson", "spearman"]:
                    r, p, n = _safe_corr(ret1, future_news, method)
                    rows.append({
                        "coin": coin,
                        "direction": "return_to_future_news",
                        "feature": feature_name,
                        "horizon_days": lead,
                        "method": method,
                        "correlation": r,
                        "p_value": p,
                        "n": n,
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        valid = out["p_value"].notna()
        out["p_value_fdr"] = np.nan
        out["significant_fdr_5pct"] = False
        if valid.any():
            rejected, corrected, _, _ = multipletests(
                out.loc[valid, "p_value"].values,
                alpha=0.05,
                method="fdr_bh",
            )
            out.loc[valid, "p_value_fdr"] = corrected
            out.loc[valid, "significant_fdr_5pct"] = rejected
    return out


def granger_study(df: pd.DataFrame, maxlag: int = 7) -> pd.DataFrame:
    """
    Test op dagelijkse data of nieuws extra lag-informatie bevat voor 1-daagse returns
    en ook de omgekeerde richting. We gebruiken stationairdere afgeleide variabelen.
    """
    rows = []

    for coin in COINS:
        news_vars = {
            "tone_change1": df[f"{coin}_news_tone_change1"],
            "volume_change1": df[f"{coin}_news_volume_change1"],
            "negative_share": df[f"{coin}_news_negative_share"],
            "positive_share": df[f"{coin}_news_positive_share"],
        }
        ret = df[f"{coin}_ret1"]

        for fname, news in news_vars.items():
            for direction in ["news_to_return", "return_to_news"]:
                if direction == "news_to_return":
                    pair = pd.concat([ret, news], axis=1).dropna()
                else:
                    pair = pd.concat([news, ret], axis=1).dropna()

                if len(pair) < 500:
                    continue

                # Winsorize om incidenten/extreme outliers niet alle regressie te laten dragen.
                pair = pair.clip(pair.quantile(0.005), pair.quantile(0.995), axis=1)

                try:
                    res = grangercausalitytests(pair.values, maxlag=maxlag, verbose=False)
                    for lag, obj in res.items():
                        p = obj[0]["ssr_ftest"][1]
                        rows.append({
                            "coin": coin,
                            "direction": direction,
                            "feature": fname,
                            "lag_days": int(lag),
                            "p_value": float(p),
                            "n": len(pair),
                        })
                except Exception:
                    continue

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_fdr"] = np.nan
        out["significant_fdr_5pct"] = False
        rejected, corrected, _, _ = multipletests(
            out["p_value"].values, alpha=0.05, method="fdr_bh"
        )
        out["p_value_fdr"] = corrected
        out["significant_fdr_5pct"] = rejected
    return out


def event_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vergelijk toekomstige returns na extreme nieuws-/sentimentdagen met de overige dagen.
    Extreme = onderste/bovenste deciel binnen de beschikbare historie.
    """
    rows = []
    for coin in COINS:
        features = {
            "tone": df[f"{coin}_news_tone"],
            "negative_share": df[f"{coin}_news_negative_share"],
            "positive_share": df[f"{coin}_news_positive_share"],
            "volume_z30": df[f"{coin}_news_volume_z30"],
        }

        for fname, s in features.items():
            z = s.dropna()
            if len(z) < 500:
                continue
            q10, q90 = z.quantile([0.10, 0.90])

            groups = {
                "low_decile": s <= q10,
                "high_decile": s >= q90,
            }
            for h in [1, 7, 30]:
                future_ret = df[f"{coin}_close"].shift(-h) / df[f"{coin}_close"] - 1
                overall = future_ret.mean()

                for gname, mask in groups.items():
                    vals = future_ret[mask].dropna()
                    rows.append({
                        "coin": coin,
                        "feature": fname,
                        "group": gname,
                        "horizon_days": h,
                        "n": len(vals),
                        "avg_future_return": vals.mean() if len(vals) else np.nan,
                        "median_future_return": vals.median() if len(vals) else np.nan,
                        "overall_avg_return": overall,
                        "difference_vs_all": (
                            vals.mean() - overall if len(vals) else np.nan
                        ),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modellen
# ---------------------------------------------------------------------------

def make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=18,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def make_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=18,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )


def walk_forward(
    df: pd.DataFrame,
    coin: str,
    horizon: int,
    features: List[str],
    allowed_index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    y_class = f"{coin}_target_up_{horizon}"
    y_ret = f"{coin}_target_ret_{horizon}"

    usable = df.dropna(subset=features + [y_class, y_ret]).copy()
    if allowed_index is not None:
        usable = usable.loc[usable.index.intersection(allowed_index)]

    if usable.empty:
        raise RuntimeError("Te weinig bruikbare data voor backtest.")

    start_test = usable.index.min() + pd.DateOffset(years=MIN_TRAIN_YEARS)
    test_dates = usable.index[usable.index >= start_test]

    rows = []
    i = 0
    while i < len(test_dates):
        block_dates = test_dates[i:i + RETRAIN_EVERY_DAYS]
        pred_start = block_dates[0]

        # Anti-lookahead voor het target zelf.
        label_cutoff = pred_start - pd.Timedelta(days=horizon)
        train = usable.loc[usable.index <= label_cutoff]
        test = usable.loc[block_dates]

        if len(train) < 500 or test.empty:
            i += RETRAIN_EVERY_DAYS
            continue

        clf = make_classifier()
        reg = make_regressor()
        clf.fit(train[features], train[y_class].astype(int))
        reg.fit(train[features], train[y_ret])

        prob = clf.predict_proba(test[features])[:, 1]
        pred_ret = reg.predict(test[features])

        for dt, p, pr, actual_up, actual_ret in zip(
            test.index, prob, pred_ret, test[y_class], test[y_ret]
        ):
            rows.append({
                "date": dt,
                "prob_up": float(p),
                "pred_ret": float(pr),
                "actual_up": int(actual_up),
                "actual_ret": float(actual_ret),
            })

        i += RETRAIN_EVERY_DAYS

    return pd.DataFrame(rows).set_index("date").sort_index()


def evaluate(bt: pd.DataFrame) -> Dict[str, float]:
    if bt.empty:
        return {}

    pred_up = (bt["prob_up"] >= 0.5).astype(int)
    baseline_class = int(bt["actual_up"].mean() >= 0.5)
    baseline_pred = np.full(len(bt), baseline_class)

    out = {
        "n": len(bt),
        "accuracy": accuracy_score(bt["actual_up"], pred_up),
        "balanced_accuracy": balanced_accuracy_score(bt["actual_up"], pred_up),
        "baseline_accuracy": accuracy_score(bt["actual_up"], baseline_pred),
        "brier": brier_score_loss(bt["actual_up"], bt["prob_up"]),
        "log_loss": log_loss(bt["actual_up"], np.clip(bt["prob_up"], 1e-6, 1 - 1e-6)),
        "mae_return": mean_absolute_error(bt["actual_ret"], bt["pred_ret"]),
        "actual_up_rate": bt["actual_up"].mean(),
    }

    out["auc"] = (
        roc_auc_score(bt["actual_up"], bt["prob_up"])
        if bt["actual_up"].nunique() > 1 else np.nan
    )

    position = (bt["prob_up"] >= 0.55).astype(int)
    changes = position.diff().abs().fillna(position.iloc[0])
    strategy_ret = position * bt["actual_ret"] - changes * TRADING_COST
    out["avg_strategy_horizon_ret"] = strategy_ret.mean()
    out["avg_buyhold_horizon_ret"] = bt["actual_ret"].mean()

    return out


def paired_model_comparison(
    base_bt: pd.DataFrame,
    news_bt: pd.DataFrame,
    coin: str,
    horizon: int,
) -> Dict[str, float]:
    z = base_bt.add_prefix("base_").join(news_bt.add_prefix("news_"), how="inner")
    if z.empty:
        return {}

    # Zelfde out-of-sample observaties, dus een eerlijke vergelijking.
    y = z["base_actual_up"].astype(int)

    base_brier = np.mean((z["base_prob_up"] - y) ** 2)
    news_brier = np.mean((z["news_prob_up"] - y) ** 2)

    base_logloss = log_loss(y, np.clip(z["base_prob_up"], 1e-6, 1 - 1e-6))
    news_logloss = log_loss(y, np.clip(z["news_prob_up"], 1e-6, 1 - 1e-6))

    base_auc = roc_auc_score(y, z["base_prob_up"]) if y.nunique() > 1 else np.nan
    news_auc = roc_auc_score(y, z["news_prob_up"]) if y.nunique() > 1 else np.nan

    base_acc = accuracy_score(y, (z["base_prob_up"] >= 0.5).astype(int))
    news_acc = accuracy_score(y, (z["news_prob_up"] >= 0.5).astype(int))

    base_mae = mean_absolute_error(z["base_actual_ret"], z["base_pred_ret"])
    news_mae = mean_absolute_error(z["news_actual_ret"], z["news_pred_ret"])

    # Simpele block bootstrap op het verschil in Brier-score.
    # Negatief verschil (news-base) is beter voor het nieuwsmodel.
    losses_diff = (
        (z["news_prob_up"] - y) ** 2
        - (z["base_prob_up"] - y) ** 2
    ).to_numpy()

    rng = np.random.default_rng(42)
    block = 30
    boots = []
    n = len(losses_diff)
    if n >= 200:
        for _ in range(1000):
            sampled = []
            while len(sampled) < n:
                s = rng.integers(0, max(1, n - block + 1))
                sampled.extend(losses_diff[s:s + block].tolist())
            boots.append(np.mean(sampled[:n]))
        ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
    else:
        ci_low, ci_high = np.nan, np.nan

    return {
        "coin": coin,
        "horizon_days": horizon,
        "n_common": len(z),
        "base_auc": base_auc,
        "news_auc": news_auc,
        "delta_auc": news_auc - base_auc if not np.isnan(base_auc) else np.nan,
        "base_accuracy": base_acc,
        "news_accuracy": news_acc,
        "delta_accuracy": news_acc - base_acc,
        "base_brier": base_brier,
        "news_brier": news_brier,
        "delta_brier_news_minus_base": news_brier - base_brier,
        "brier_delta_ci95_low": ci_low,
        "brier_delta_ci95_high": ci_high,
        "base_logloss": base_logloss,
        "news_logloss": news_logloss,
        "delta_logloss": news_logloss - base_logloss,
        "base_mae_return": base_mae,
        "news_mae_return": news_mae,
        "delta_mae_return": news_mae - base_mae,
    }


def latest_prediction(
    df: pd.DataFrame,
    coin: str,
    horizon: int,
    features: List[str],
    backtest: pd.DataFrame,
) -> Dict[str, float]:
    y_class = f"{coin}_target_up_{horizon}"
    y_ret = f"{coin}_target_ret_{horizon}"

    latest_date = df.dropna(subset=features).index.max()
    latest_x = df.loc[[latest_date], features]

    label_cutoff = latest_date - pd.Timedelta(days=horizon)
    train = df.loc[df.index <= label_cutoff].dropna(
        subset=features + [y_class, y_ret]
    )

    clf = make_classifier()
    reg = make_regressor()
    clf.fit(train[features], train[y_class].astype(int))
    reg.fit(train[features], train[y_ret])

    p_up = clf.predict_proba(latest_x)[:, 1][0]
    p_ret = reg.predict(latest_x)[0]

    residuals = backtest["actual_ret"] - backtest["pred_ret"]
    q10, q90 = residuals.quantile([0.10, 0.90])

    return {
        "date": latest_date,
        "prob_up": float(p_up),
        "expected_return": float(p_ret),
        "range80_low": float(p_ret + q10),
        "range80_high": float(p_ret + q90),
    }


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n.v.t."
    return f"{100*x:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def news_edge_label(row: Dict[str, float]) -> str:
    """
    Conservatieve rule-of-thumb:
    - hogere AUC,
    - lagere Brier,
    - lagere log-loss,
    - en bootstrap-CI van Brier-delta bij voorkeur volledig onder nul.
    """
    if not row:
        return "ONBEKEND"
    improves_three = (
        row["delta_auc"] > 0
        and row["delta_brier_news_minus_base"] < 0
        and row["delta_logloss"] < 0
    )
    ci_confirm = (
        not np.isnan(row["brier_delta_ci95_high"])
        and row["brier_delta_ci95_high"] < 0
    )

    if improves_three and ci_confirm:
        return "ROBUSTE INCREMENTELE NIEUWS-EDGE"
    if improves_three:
        return "MOGELIJKE NIEUWS-EDGE"
    return "GEEN CONSISTENTE NIEUWS-EDGE"


def summarize_relationships(corr: pd.DataFrame, granger: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("NIEUWS ↔ KOERS: RELATIE-ONDERZOEK")
    print("=" * 74)

    if corr.empty:
        print("Geen correlatieresultaten beschikbaar.")
        return

    for coin in COINS:
        z = corr[
            (corr["coin"] == coin)
            & (corr["direction"] == "news_to_future_return")
            & (corr["method"] == "spearman")
        ].copy()
        z = z.sort_values("correlation", key=lambda s: s.abs(), ascending=False).head(6)

        print(f"\n{coin}: sterkste lead-correlaties (Spearman, nieuws -> toekomstig rendement)")
        for _, r in z.iterrows():
            sig = "*" if bool(r.get("significant_fdr_5pct", False)) else ""
            print(
                f"  {r['feature']:24s} {int(r['horizon_days']):2d}d  "
                f"rho={r['correlation']:+.3f}  FDR-p={r['p_value_fdr']:.4f}{sig}"
            )

        rev = corr[
            (corr["coin"] == coin)
            & (corr["direction"] == "return_to_future_news")
            & (corr["method"] == "spearman")
        ]
        if not rev.empty:
            strongest = rev.iloc[rev["correlation"].abs().argmax()]
            print(
                f"  Sterkste reverse relatie: koers -> {strongest['feature']} "
                f"({int(strongest['horizon_days'])}d), "
                f"rho={strongest['correlation']:+.3f}, "
                f"FDR-p={strongest['p_value_fdr']:.4f}"
            )

    if not granger.empty:
        sig = granger[granger["significant_fdr_5pct"]]
        print(f"\nGranger-tests na FDR-correctie: {len(sig)} significante lag-relaties.")
        if len(sig):
            print(sig.sort_values("p_value_fdr").head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-news-refresh", action="store_true")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("BTC/ETH Predictor v2 — koers + nieuws")
    print("=" * 74)
    print("1/5 Koersdata downloaden...")
    price = download_price_data()
    df = add_price_features(price)

    news_start = max(price.index.min(), GDELT_START)
    # Alleen VOLLEDIGE vorige kalenderdagen als nieuwsfeature gebruiken.
    news_end = pd.Timestamp(date.today() - timedelta(days=1))

    print("2/5 Historische GDELT-nieuwsdata laden/updaten...")
    print(
        f"    Gratis GDELT DOC-historie: {news_start.date()} t/m {news_end.date()} "
        f"(GDELT DOC start 01-01-2017)"
    )
    raw_news = load_or_update_news(
        cache_dir=cache_dir,
        start=news_start,
        end=news_end,
        force=args.force_news_refresh,
    )
    raw_news.to_csv(output_dir / "raw_news_daily.csv", index_label="date")

    print("3/5 Nieuwsfeatures construeren en relaties testen...")
    df = add_news_features(df, raw_news)

    corr = correlation_study(df)
    corr.to_csv(output_dir / "news_correlations.csv", index=False)

    granger = granger_study(df)
    granger.to_csv(output_dir / "granger_tests.csv", index=False)

    events = event_study(df)
    events.to_csv(output_dir / "news_event_study.csv", index=False)

    summarize_relationships(corr, granger)

    base_features = base_feature_columns(df)
    nfeatures = news_feature_columns(df)
    combined_features = base_features + nfeatures

    comparisons = []
    latest_rows = []

    print("\n4/5 Walk-forward vergelijking: koers-only versus koers + nieuws...")
    for coin in COINS:
        for h in HORIZONS:
            y_class = f"{coin}_target_up_{h}"
            y_ret = f"{coin}_target_ret_{h}"

            # De twee modellen krijgen exact dezelfde datums.
            fair = df.dropna(subset=combined_features + [y_class, y_ret])
            if fair.empty:
                raise RuntimeError(
                    "Te weinig complete prijs+nieuwsdata. Controleer de GDELT-cache/API."
                )
            allowed = fair.index

            print(f"\n{coin} — horizon {h} dagen")
            base_bt = walk_forward(df, coin, h, base_features, allowed_index=allowed)
            news_bt = walk_forward(df, coin, h, combined_features, allowed_index=allowed)

            base_bt.to_csv(output_dir / f"{coin}_{h}d_backtest_price_only.csv")
            news_bt.to_csv(output_dir / f"{coin}_{h}d_backtest_price_news.csv")

            b = evaluate(base_bt)
            n = evaluate(news_bt)
            comp = paired_model_comparison(base_bt, news_bt, coin, h)
            comp["assessment"] = news_edge_label(comp)
            comparisons.append(comp)

            print(f"  Prijs-only AUC         : {b['auc']:.3f}")
            print(f"  Prijs+nieuws AUC       : {n['auc']:.3f}")
            print(f"  Δ AUC                  : {comp['delta_auc']:+.3f}")
            print(f"  Prijs-only Brier       : {b['brier']:.4f}")
            print(f"  Prijs+nieuws Brier     : {n['brier']:.4f}")
            print(
                f"  Δ Brier (nieuws-base) : "
                f"{comp['delta_brier_news_minus_base']:+.4f} (lager is beter)"
            )
            print(f"  Beoordeling            : {comp['assessment']}")

            lp_base = latest_prediction(df, coin, h, base_features, base_bt)
            lp_news = latest_prediction(df, coin, h, combined_features, news_bt)
            latest_rows.extend([
                {
                    "coin": coin,
                    "horizon_days": h,
                    "model": "price_only",
                    **lp_base,
                },
                {
                    "coin": coin,
                    "horizon_days": h,
                    "model": "price_plus_news",
                    **lp_news,
                },
            ])

            print(
                f"  Actuele kans omhoog (prijs+nieuws): "
                f"{pct(lp_news['prob_up'])}"
            )
            print(
                f"  Verwachte return (prijs+nieuws)   : "
                f"{pct(lp_news['expected_return'])}"
            )
            print(
                f"  Empirische 80%-band               : "
                f"{pct(lp_news['range80_low'])} tot {pct(lp_news['range80_high'])}"
            )

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_dir / "model_comparison_price_vs_news.csv", index=False)

    latest_df = pd.DataFrame(latest_rows)
    latest_df.to_csv(output_dir / "latest_predictions.csv", index=False)

    print("\n5/5 Klaar.")
    print(f"Resultaten staan in: {output_dir.resolve()}")
    print("\nInterpretatie:")
    print("- Een significante correlatie is niet automatisch voorspellend.")
    print("- Reverse tests laten zien of nieuws mogelijk vooral OP koers reageert.")
    print("- De belangrijkste toets is of prijs+nieuws out-of-sample beter is dan prijs-only.")
    print("- FDR-correctie beperkt false positives door de vele correlatietests.")
    print("- Een nieuws-edge die slechts in één metric zichtbaar is, wordt niet als robuust gezien.")


if __name__ == "__main__":
    main()
