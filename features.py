from __future__ import annotations
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from config import Config

HALVINGS = pd.to_datetime(["2012-11-28","2016-07-09","2020-05-11","2024-04-20"])

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def _rolling_z(s, n, minp=None):
    minp = minp or max(10, n//2)
    return (s - s.rolling(n, min_periods=minp).mean()) / s.rolling(n, min_periods=minp).std().replace(0,np.nan)

def _add_transforms(x: pd.DataFrame, cols: List[str], prefix_group: str):
    for c in cols:
        s = pd.to_numeric(x[c], errors="coerce")
        # Ratios/percentages can contain negatives; log1p only for strictly non-negative levels.
        if s.dropna().empty:
            continue
        if s.dropna().min() >= 0 and s.quantile(0.99) > 10:
            ls = np.log1p(s)
            x[f"{c}_log"] = ls
            x[f"{c}_logchg7"] = ls.diff(7)
            x[f"{c}_logchg30"] = ls.diff(30)
            x[f"{c}_z90"] = _rolling_z(ls, 90)
        else:
            x[f"{c}_chg1"] = s.diff()
            x[f"{c}_chg7"] = s.diff(7)
            x[f"{c}_z90"] = _rolling_z(s, 90)
    return x

def build_features(sources: Dict[str,pd.DataFrame], cfg: Config) -> pd.DataFrame:
    crypto = sources["crypto"].copy()
    idx = crypto.index
    x = crypto.copy()

    for name, d in sources.items():
        if name == "crypto" or d is None or d.empty:
            continue
        x = x.join(d.reindex(idx), how="left")

    # Forward-fill exogenous state variables, but not raw crypto OHLCV.
    exog = [c for c in x.columns if not c.startswith(("BTC_","ETH_"))]
    x[exog] = x[exog].ffill(limit=45)

    # Technical / microstructure proxies.
    for coin in cfg.coins:
        p = x[f"{coin}_close"]
        high = x[f"{coin}_high"]
        low = x[f"{coin}_low"]
        op = x[f"{coin}_open"]
        vol = x[f"{coin}_volume"].replace(0,np.nan)
        lr = np.log(p).diff()
        x[f"TECH_{coin}_ret1"] = lr
        x[f"TECH_{coin}_gap"] = op/p.shift(1)-1
        x[f"TECH_{coin}_intraday"] = p/op-1
        x[f"TECH_{coin}_range"] = (high-low)/p
        x[f"TECH_{coin}_rsi14"] = rsi(p)/100
        for n in [2,3,7,14,30,60,90,180,365]:
            x[f"TECH_{coin}_mom{n}"] = p.pct_change(n)
        for n in [7,14,30,60,90]:
            rv = lr.rolling(n).std()*np.sqrt(365)
            x[f"TECH_{coin}_vol{n}"] = rv
            neg = lr.where(lr<0,0).rolling(n).std()*np.sqrt(365)
            x[f"TECH_{coin}_downvol{n}"] = neg
        for n in [7,14,30,90,200,365]:
            ma = p.rolling(n).mean()
            x[f"TECH_{coin}_ma{n}"] = p/ma-1
        for n in [20,60,120]:
            m = lr.rolling(n).mean()
            sd = lr.rolling(n).std()
            x[f"TECH_{coin}_retz{n}"] = (lr-m)/sd.replace(0,np.nan)
            x[f"TECH_{coin}_skew{n}"] = lr.rolling(n).skew()
            x[f"TECH_{coin}_kurt{n}"] = lr.rolling(n).kurt()
        x[f"TECH_{coin}_volz30"] = _rolling_z(np.log1p(vol),30)
        x[f"TECH_{coin}_volz90"] = _rolling_z(np.log1p(vol),90)
        x[f"TECH_{coin}_dd30"] = p/p.rolling(30).max()-1
        x[f"TECH_{coin}_dd90"] = p/p.rolling(90).max()-1
        x[f"TECH_{coin}_dd365"] = p/p.rolling(365).max()-1
        # Bollinger-like normalized distance
        x[f"TECH_{coin}_boll20"] = (p-p.rolling(20).mean())/p.rolling(20).std().replace(0,np.nan)
        # simple illiquidity proxy
        x[f"TECH_{coin}_amihud"] = lr.abs()/(p*vol).replace(0,np.nan)

        for h in cfg.horizons:
            future = p.shift(-h)/p-1
            x[f"TARGET_{coin}_ret_{h}"] = future
            x[f"TARGET_{coin}_up_{h}"] = np.where(future.notna(), (future>0).astype(float), np.nan)

    # BTC/ETH cross-relationship
    x["CROSS_eth_btc"] = x["ETH_close"]/x["BTC_close"]
    x["CROSS_eth_btc_mom7"] = x["CROSS_eth_btc"].pct_change(7)
    x["CROSS_eth_btc_mom30"] = x["CROSS_eth_btc"].pct_change(30)
    x["CROSS_ret_spread1"] = x["TECH_BTC_ret1"]-x["TECH_ETH_ret1"]
    x["CROSS_corr30"] = x["TECH_BTC_ret1"].rolling(30).corr(x["TECH_ETH_ret1"])
    x["CROSS_corr90"] = x["TECH_BTC_ret1"].rolling(90).corr(x["TECH_ETH_ret1"])

    # Cross-asset market features (data source has already been safely lagged below).
    cross_cols = [c for c in x.columns if c.startswith("X_") and c.endswith("_close")]
    for c in cross_cols:
        s = x[c].shift(cfg.exogenous_lag_days)
        x[c] = s
        base = c[:-6]
        x[f"{base}_ret1"] = np.log(s).diff()
        x[f"{base}_mom7"] = s.pct_change(7)
        x[f"{base}_mom30"] = s.pct_change(30)
        x[f"{base}_vol30"] = np.log(s).diff().rolling(30).std()*np.sqrt(252)
        x[f"{base}_z90"] = _rolling_z(s,90)
        x[f"{base}_btc_corr60"] = x[f"{base}_ret1"].rolling(60).corr(x["TECH_BTC_ret1"])

    # Transform macro/on-chain/funding/sentiment/ETF/trends/custom raw fields.
    groups = {
        "MACRO": [c for c in x if c.startswith("FRED_")],
        "ONCHAIN": [c for c in x if c.startswith(("OC_","STABLE_"))],
        "DERIV": [c for c in x if c.startswith(("DERIV_","OPTIONS_"))],
        "SENT": [c for c in x if c.startswith(("FGI_","TRENDS_"))],
        "ETF": [c for c in x if c.startswith("ETF_")],
        "CUSTOM": [c for c in x if c.startswith("CUSTOM_")],
    }
    for group, cols in groups.items():
        x = _add_transforms(x, cols, group)

    # ETF cumulative pressure
    for c in [c for c in x if c.endswith("_netflow_usdm")]:
        x[f"{c}_sum5"] = x[c].rolling(5,min_periods=1).sum()
        x[f"{c}_sum20"] = x[c].rolling(20,min_periods=1).sum()
        x[f"{c}_z60"] = _rolling_z(x[c],60,20)

    # Fear & Greed specifically
    if "FGI_value" in x:
        x["SENT_fgi_chg1"] = x["FGI_value"].diff()
        x["SENT_fgi_chg7"] = x["FGI_value"].diff(7)
        x["SENT_fgi_z90"] = _rolling_z(x["FGI_value"],90)

    # Derivatives specifically
    for coin in cfg.coins:
        c = f"DERIV_{coin}_funding_sum"
        if c in x:
            x[f"DERIV_{coin}_funding_sum7"] = x[c].rolling(7,min_periods=1).sum()
            x[f"DERIV_{coin}_funding_z90"] = _rolling_z(x[c],90)

    # Halving-cycle features: only time since a known past event, never time to a future actual date.
    dates = pd.Series(x.index, index=x.index)
    for coin in ["BTC"]:
        last_halving = []
        for dt in x.index:
            past = HALVINGS[HALVINGS <= dt]
            last_halving.append(past.max() if len(past) else pd.NaT)
        last_halving = pd.Series(last_halving,index=x.index)
        x["REGIME_BTC_days_since_halving"] = (pd.Series(x.index,index=x.index)-last_halving).dt.days
        x["REGIME_BTC_halving_sin"] = np.sin(2*np.pi*x["REGIME_BTC_days_since_halving"]/1460.0)
        x["REGIME_BTC_halving_cos"] = np.cos(2*np.pi*x["REGIME_BTC_days_since_halving"]/1460.0)

    # Market regimes.
    x["REGIME_BTC_trend200"] = (x["BTC_close"] > x["BTC_close"].rolling(200).mean()).astype(float)
    x["REGIME_BTC_vol_pct"] = x["TECH_BTC_vol30"].rolling(730,min_periods=180).rank(pct=True)
    x["REGIME_ETH_trend200"] = (x["ETH_close"] > x["ETH_close"].rolling(200).mean()).astype(float)
    if "X_VIX_close" in x:
        x["REGIME_riskstress"] = _rolling_z(x["X_VIX_close"],252,100)
    if "FRED_STLFSI4" in x:
        x["REGIME_financial_stress"] = _rolling_z(x["FRED_STLFSI4"],104,40)

    # Calendar features.
    dow = pd.Series(x.index.dayofweek,index=x.index)
    doy = pd.Series(x.index.dayofyear,index=x.index)
    x["CAL_dow_sin"] = np.sin(2*np.pi*dow/7)
    x["CAL_dow_cos"] = np.cos(2*np.pi*dow/7)
    x["CAL_year_sin"] = np.sin(2*np.pi*doy/365.25)
    x["CAL_year_cos"] = np.cos(2*np.pi*doy/365.25)

    return x.replace([np.inf,-np.inf],np.nan)

def add_news_frame(x: pd.DataFrame, news_features: pd.DataFrame) -> pd.DataFrame:
    if news_features is None or news_features.empty:
        return x
    cols = [c for c in news_features.columns if c not in x.columns]
    return x.join(news_features[cols], how="left")

def feature_groups(df: pd.DataFrame) -> Dict[str,List[str]]:
    all_cols = list(df.columns)
    # Absolute raw OHLCV levels are deliberately excluded from model features.
    # The engineered stationary/relative features carry the useful information
    # without letting the model "predict" from the mere passage of time/price level.
    groups = {
        "technical": [c for c in all_cols if c.startswith(("TECH_","CROSS_","REGIME_","CAL_"))],
        "cross_macro": [c for c in all_cols if c.startswith(("X_","FRED_"))],
        "onchain": [c for c in all_cols if c.startswith(("OC_","STABLE_"))],
        "sentiment": [c for c in all_cols if c.startswith(("SENT_","FGI_","TRENDS_"))],
        "derivatives": [c for c in all_cols if c.startswith(("DERIV_","OPTIONS_"))],
        "etf": [c for c in all_cols if c.startswith("ETF_")],
        "news": [c for c in all_cols if "_news_" in c],
        "custom": [c for c in all_cols if c.startswith("CUSTOM_")],
    }
    # Remove targets and duplicated columns.
    for k,v in groups.items():
        seen=[]
        for c in v:
            if c in df.columns and "TARGET_" not in c and c not in seen:
                seen.append(c)
        groups[k]=seen
    return groups

def build_experts(df: pd.DataFrame) -> Dict[str,List[str]]:
    g = feature_groups(df)
    core = g["technical"] + g["cross_macro"]
    experts = {"core_long": core}
    if g["onchain"]:
        experts["onchain_long"] = core + g["onchain"]
    if g["news"]:
        experts["news_mid"] = core + g["news"]
    if g["sentiment"] or g["derivatives"]:
        experts["sentiment_deriv"] = core + g["sentiment"] + g["derivatives"]
    if g["onchain"] or g["news"] or g["sentiment"] or g["derivatives"]:
        experts["all_fundamentals"] = core + g["onchain"] + g["news"] + g["sentiment"] + g["derivatives"]
    if g["etf"]:
        experts["institutional_recent"] = core + g["etf"]
        experts["all_recent"] = core + g["onchain"] + g["news"] + g["sentiment"] + g["derivatives"] + g["etf"] + g["custom"]
    elif g["custom"]:
        experts["custom_recent"] = core + g["custom"]
    return {k:list(dict.fromkeys(v)) for k,v in experts.items() if v}
