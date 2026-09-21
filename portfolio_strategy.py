from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import trading_strategy as ts

COINS = ["BTC", "ETH"]
QUALITY_LOOKBACK = 180
QUALITY_MIN_PERIODS = 60
VOL_LOOKBACK = 30
TRANSACTION_COST = ts.TRANSACTION_COST
ANNUAL_TARGET = ts.ANNUAL_TARGET

# Totale portefeuilleblootstelling. De strategie kiest alleen uit deze niveaus.
EXPOSURE_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]


def _clean(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _read_meta(output_dir: Path, coin: str, horizon: int) -> pd.DataFrame:
    p = output_dir / f"{coin}_{horizon}d_META_oos.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if "date" not in d.columns:
        d = d.rename(columns={d.columns[0]: "date"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    d = d.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    keep = [c for c in ["prob_up", "pred_ret", "actual_ret", "actual_up"] if c in d.columns]
    return d[keep]


def _rolling_quality(d: pd.DataFrame) -> pd.Series:
    """
    Alleen verleden telt mee. Kwaliteit is gebaseerd op rolling Brier skill
    versus een 50/50-baseline (Brier 0.25). De huidige observatie wordt
    verschoven zodat actual_ret van dezelfde dag nooit de positie bepaalt.
    """
    if d.empty or "prob_up" not in d.columns:
        return pd.Series(dtype=float)

    if "actual_up" in d.columns:
        actual = pd.to_numeric(d["actual_up"], errors="coerce")
    elif "actual_ret" in d.columns:
        actual = (pd.to_numeric(d["actual_ret"], errors="coerce") > 0).astype(float)
    else:
        return pd.Series(0.5, index=d.index)

    prob = pd.to_numeric(d["prob_up"], errors="coerce")
    brier = (prob - actual) ** 2
    hist = brier.shift(1).rolling(QUALITY_LOOKBACK, min_periods=QUALITY_MIN_PERIODS).mean()

    # 0.5 = neutraal; beter dan baseline loopt richting 1.0; slechter richting 0.25.
    skill = (0.25 - hist) / 0.25
    q = (0.5 + 0.5 * skill).clip(lower=0.25, upper=1.0)
    return q.fillna(0.5)


def _horizon_component(prob_up: float, pred_ret: float, horizon: int) -> float:
    direction = np.clip(2.0 * prob_up - 1.0, -1.0, 1.0)
    ret_component = math.tanh(pred_ret / ts.RETURN_SCALES[horizon])
    return float(0.60 * direction + 0.40 * ret_component)


def _coin_history(output_dir: Path, coin: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    all_dates: set[pd.Timestamp] = set()
    raw: dict[int, pd.DataFrame] = {}

    for h in ts.HORIZONS:
        d = _read_meta(output_dir, coin, h)
        if d.empty:
            continue
        raw[h] = d
        all_dates.update(d.index.tolist())

    if not raw or not all_dates:
        return pd.DataFrame()

    index = pd.DatetimeIndex(sorted(all_dates))
    result = pd.DataFrame(index=index)
    weighted_score = pd.Series(0.0, index=index)
    weighted_quality = pd.Series(0.0, index=index)
    weighted_conf = pd.Series(0.0, index=index)
    denom = pd.Series(0.0, index=index)

    for h, base_w in ts.HORIZON_WEIGHTS.items():
        d = raw.get(h)
        if d is None or d.empty:
            continue

        q = _rolling_quality(d)
        p = pd.to_numeric(d.get("prob_up"), errors="coerce")
        r = pd.to_numeric(d.get("pred_ret"), errors="coerce")
        comp = pd.Series(index=d.index, dtype=float)
        valid = p.notna() & r.notna()
        comp.loc[valid] = [
            _horizon_component(float(pp), float(rr), h)
            for pp, rr in zip(p.loc[valid], r.loc[valid])
        ]
        conf = (2.0 * (p - 0.5).abs()).clip(0.0, 1.0)

        aligned_comp = comp.reindex(index)
        aligned_q = q.reindex(index)
        aligned_conf = conf.reindex(index)
        valid2 = aligned_comp.notna() & aligned_q.notna()
        dynamic_w = base_w * aligned_q.where(valid2, 0.0)

        weighted_score += dynamic_w * aligned_comp.fillna(0.0)
        weighted_quality += base_w * aligned_q.where(valid2, 0.0)
        weighted_conf += base_w * aligned_conf.where(valid2, 0.0)
        denom += base_w * valid2.astype(float)

    result["raw_score"] = (weighted_score / weighted_quality.replace(0, np.nan)).clip(-1, 1)
    result["quality"] = (weighted_quality / denom.replace(0, np.nan)).clip(0.25, 1.0)
    result["confidence"] = (weighted_conf / denom.replace(0, np.nan)).clip(0.0, 1.0)

    # Kwaliteit en confidence beïnvloeden de kapitaalsterkte, maar mogen het
    # richtinggevende model niet volledig domineren.
    result["effective_score"] = (
        result["raw_score"]
        * (0.60 + 0.40 * result["quality"])
        * (0.60 + 0.40 * result["confidence"])
    ).clip(-1, 1)

    one = raw.get(1, pd.DataFrame())
    if not one.empty and "actual_ret" in one.columns:
        result["next_return"] = pd.to_numeric(one["actual_ret"], errors="coerce").reindex(index)

    return result.dropna(subset=["effective_score"])


def _regime_series(btc: pd.DataFrame) -> pd.DataFrame:
    if btc.empty or "next_return" not in btc.columns:
        return pd.DataFrame(index=btc.index)

    # pseudo_price op datum t bevat alleen rendementen t-1 en ouder.
    realized = pd.to_numeric(btc["next_return"], errors="coerce")
    pseudo = (1.0 + realized.shift(1).fillna(0.0)).cumprod()
    ma50 = pseudo.rolling(50, min_periods=30).mean()
    ma200 = pseudo.rolling(200, min_periods=120).mean()
    vol30 = realized.shift(1).rolling(VOL_LOOKBACK, min_periods=20).std() * np.sqrt(365.25)

    out = pd.DataFrame(index=btc.index)
    out["price"] = pseudo
    out["ma50"] = ma50
    out["ma200"] = ma200
    out["vol30"] = vol30

    labels = []
    caps = []
    for dt in out.index:
        p = out.at[dt, "price"]
        m50 = out.at[dt, "ma50"]
        m200 = out.at[dt, "ma200"]
        vol = out.at[dt, "vol30"]

        if pd.isna(m200):
            label, cap = "onvoldoende historie", 0.50
        elif p > m200 and m50 > m200:
            label, cap = "bull", 1.00
        elif p < m200 and m50 < m200:
            label, cap = "bear", 0.50
        else:
            label, cap = "neutraal", 0.75

        if not pd.isna(vol) and vol > 0.80:
            label += " / hoge volatiliteit"
            cap = max(0.25, cap - 0.25)

        labels.append(label)
        caps.append(cap)

    out["regime"] = labels
    out["exposure_cap"] = caps
    return out


def _desired_total_exposure(best_score: float, cap: float) -> float:
    if best_score < 0.04:
        raw = 0.0
    elif best_score < 0.10:
        raw = 0.25
    elif best_score < 0.18:
        raw = 0.50
    elif best_score < 0.28:
        raw = 0.75
    else:
        raw = 1.00
    return min(raw, cap)


def _allocate(btc_score: float, eth_score: float, cap: float) -> tuple[float, float, float]:
    positives = {
        "BTC": max(0.0, btc_score),
        "ETH": max(0.0, eth_score),
    }
    best = max(positives.values())
    total = _desired_total_exposure(best, cap)
    if total <= 0:
        return 0.0, 0.0, 1.0

    # Alleen munten die ten minste 70% van het sterkste signaal halen delen mee.
    active = {k: v for k, v in positives.items() if v > 0 and v >= 0.70 * best}
    s = sum(active.values())
    if s <= 0:
        return 0.0, 0.0, 1.0

    btc_w = total * active.get("BTC", 0.0) / s
    eth_w = total * active.get("ETH", 0.0) / s
    cash_w = 1.0 - btc_w - eth_w
    return float(btc_w), float(eth_w), float(cash_w)


def _portfolio_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}

    r = frame["strategy_return"].dropna()
    b = frame["benchmark_return"].reindex(r.index).fillna(0.0)
    if r.empty:
        return {}

    eq = (1 + r).cumprod()
    beq = (1 + b).cumprod()
    years = max(len(r) / 365.25, 1 / 365.25)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    benchmark_cagr = float(beq.iloc[-1] ** (1 / years) - 1) if beq.iloc[-1] > 0 else -1.0
    dd = eq / eq.cummax() - 1
    sd = float(r.std())
    sharpe = float(r.mean() / sd * np.sqrt(365.25)) if sd > 0 else None

    turnover = frame["turnover"].fillna(0.0)
    changes = int((turnover > 1e-9).sum())

    return {
        "days": int(len(r)),
        "cagr": cagr,
        "benchmarkCagr": benchmark_cagr,
        "totalReturn": float(eq.iloc[-1] - 1),
        "benchmarkTotalReturn": float(beq.iloc[-1] - 1),
        "maxDrawdown": float(dd.min()),
        "sharpe": sharpe,
        "allocationChanges": changes,
        "averageInvested": float((frame["btc_weight"] + frame["eth_weight"]).mean()),
        "averageBTC": float(frame["btc_weight"].mean()),
        "averageETH": float(frame["eth_weight"].mean()),
        "averageCash": float(frame["cash_weight"].mean()),
    }


def backtest_portfolio(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    btc = _coin_history(output_dir, "BTC")
    eth = _coin_history(output_dir, "ETH")
    if btc.empty or eth.empty:
        return pd.DataFrame(), {}

    regime = _regime_series(btc)
    idx = btc.index.intersection(eth.index).intersection(regime.index)
    frame = pd.DataFrame(index=idx)
    frame["btc_score"] = btc["effective_score"].reindex(idx)
    frame["eth_score"] = eth["effective_score"].reindex(idx)
    frame["btc_quality"] = btc["quality"].reindex(idx)
    frame["eth_quality"] = eth["quality"].reindex(idx)
    frame["btc_confidence"] = btc["confidence"].reindex(idx)
    frame["eth_confidence"] = eth["confidence"].reindex(idx)
    frame["btc_return"] = btc["next_return"].reindex(idx)
    frame["eth_return"] = eth["next_return"].reindex(idx)
    frame["regime"] = regime["regime"].reindex(idx)
    frame["exposure_cap"] = regime["exposure_cap"].reindex(idx)
    frame = frame.dropna(subset=["btc_score", "eth_score", "btc_return", "eth_return", "exposure_cap"])

    if len(frame) < 120:
        return pd.DataFrame(), {}

    allocations = [
        _allocate(float(r.btc_score), float(r.eth_score), float(r.exposure_cap))
        for r in frame.itertuples()
    ]
    frame[["btc_weight", "eth_weight", "cash_weight"]] = allocations

    prev_btc = frame["btc_weight"].shift(1).fillna(0.0)
    prev_eth = frame["eth_weight"].shift(1).fillna(0.0)
    frame["turnover"] = (
        (frame["btc_weight"] - prev_btc).abs()
        + (frame["eth_weight"] - prev_eth).abs()
    )
    frame["strategy_return"] = (
        frame["btc_weight"] * frame["btc_return"]
        + frame["eth_weight"] * frame["eth_return"]
        - frame["turnover"] * TRANSACTION_COST
    )
    frame["benchmark_return"] = 0.5 * frame["btc_return"] + 0.5 * frame["eth_return"]
    return frame, _portfolio_metrics(frame)


def _latest_model_quality(output_dir: Path, coin: str) -> dict[int, float]:
    p = output_dir / "model_summary.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[
        d["coin"].astype(str).str.upper().eq(coin)
        & d["expert"].astype(str).str.upper().eq("META")
    ]
    out: dict[int, float] = {}
    for _, r in d.iterrows():
        try:
            h = int(r.get("horizon"))
        except Exception:
            continue
        auc = _clean(r.get("auc"))
        if auc is None:
            continue
        # AUC 0.5 -> 0.5 kwaliteit; AUC >= 0.7 -> 1.0.
        out[h] = float(np.clip(0.5 + (auc - 0.5) / 0.4, 0.25, 1.0))
    return out


def _latest_effective_score(output_dir: Path, coin: str) -> tuple[float | None, float | None, float | None]:
    latest = ts._load_latest(output_dir, coin)
    quality = _latest_model_quality(output_dir, coin)
    total = 0.0
    denom = 0.0
    qsum = 0.0
    csum = 0.0

    for h, base_w in ts.HORIZON_WEIGHTS.items():
        z = latest.get(h)
        if not z:
            continue
        p = _clean(z.get("prob_up"))
        r = _clean(z.get("pred_ret"))
        if p is None or r is None:
            continue
        q = quality.get(h, 0.5)
        conf_raw = _clean(z.get("confidence"))
        conf = float(np.clip(conf_raw if conf_raw is not None else abs(2 * p - 1), 0, 1))
        comp = _horizon_component(p, r, h)
        w = base_w * q
        total += w * comp
        denom += w
        qsum += base_w * q
        csum += base_w * conf

    if denom <= 0:
        return None, None, None

    raw = float(np.clip(total / denom, -1, 1))
    qavg = float(np.clip(qsum / sum(ts.HORIZON_WEIGHTS.values()), 0.25, 1.0))
    cavg = float(np.clip(csum / sum(ts.HORIZON_WEIGHTS.values()), 0.0, 1.0))
    effective = float(np.clip(raw * (0.60 + 0.40 * qavg) * (0.60 + 0.40 * cavg), -1, 1))
    return effective, qavg, cavg


def _recent_metrics(frame: pd.DataFrame, days: int = 365) -> dict[str, Any]:
    if frame.empty:
        return {}
    end = frame.index.max()
    recent = frame.loc[frame.index > end - pd.Timedelta(days=days)]
    return _portfolio_metrics(recent)


def _year_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
    for year, d in frame.groupby(frame.index.year):
        if len(d) < 60:
            continue
        m = _portfolio_metrics(d)
        rows.append({
            "year": int(year),
            "days": int(len(d)),
            "return": m.get("totalReturn"),
            "benchmarkReturn": m.get("benchmarkTotalReturn"),
            "maxDrawdown": m.get("maxDrawdown"),
        })
    return rows


def _cost_stress(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
    gross = frame["btc_weight"] * frame["btc_return"] + frame["eth_weight"] * frame["eth_return"]
    for cost in [0.001, 0.0025, 0.005]:
        x = frame.copy()
        x["strategy_return"] = gross - x["turnover"] * cost
        m = _portfolio_metrics(x)
        rows.append({
            "transactionCost": cost,
            "cagr": m.get("cagr"),
            "maxDrawdown": m.get("maxDrawdown"),
        })
    return rows


def build_portfolio_strategy(output_dir: Path) -> dict[str, Any]:
    frame, full = backtest_portfolio(output_dir)
    if frame.empty:
        return {
            "available": False,
            "reason": "Nog onvoldoende gezamenlijke BTC/ETH OOS-data voor de portefeuillestrategie.",
        }

    btc_score, btc_quality, btc_conf = _latest_effective_score(output_dir, "BTC")
    eth_score, eth_quality, eth_conf = _latest_effective_score(output_dir, "ETH")

    regime_hist = _regime_series(_coin_history(output_dir, "BTC"))
    if regime_hist.empty:
        current_regime = "onbekend"
        cap = 0.50
    else:
        current_regime = str(regime_hist["regime"].iloc[-1])
        cap = float(regime_hist["exposure_cap"].iloc[-1])

    if btc_score is None or eth_score is None:
        btc_w = eth_w = 0.0
        cash_w = 1.0
    else:
        btc_w, eth_w, cash_w = _allocate(btc_score, eth_score, cap)

    recent = _recent_metrics(frame, 365)
    years = _year_rows(frame)
    positive_years = sum(1 for y in years if (y.get("return") or 0) > 0)

    return {
        "available": True,
        "experimental": True,
        "recommendedAllocation": {
            "BTC": btc_w,
            "ETH": eth_w,
            "cash": cash_w,
        },
        "regime": current_regime,
        "exposureCap": cap,
        "scores": {
            "BTC": btc_score,
            "ETH": eth_score,
        },
        "quality": {
            "BTC": btc_quality,
            "ETH": eth_quality,
        },
        "confidence": {
            "BTC": btc_conf,
            "ETH": eth_conf,
        },
        "backtest": full,
        "last12Months": recent,
        "calendarYears": years,
        "positiveYears": positive_years,
        "annualTarget": ANNUAL_TARGET,
        "checks": {
            "beatsEqualWeightBuyHold": (
                full.get("cagr") is not None
                and full.get("benchmarkCagr") is not None
                and full["cagr"] > full["benchmarkCagr"]
            ),
            "maxDrawdownBelow35Pct": (
                full.get("maxDrawdown") is not None and full["maxDrawdown"] >= -0.35
            ),
            "annualTargetReached": (
                full.get("cagr") is not None and full["cagr"] >= ANNUAL_TARGET
            ),
            "majorityPositiveYears": bool(years) and positive_years > len(years) / 2,
        },
        "costStress": _cost_stress(frame),
        "method": {
            "positionLevels": EXPOSURE_LEVELS,
            "qualityLookbackDays": QUALITY_LOOKBACK,
            "regime": "BTC pseudo-price MA50/MA200 plus 30d realized volatility, all lagged.",
            "benchmark": "50/50 BTC/ETH buy & hold over dezelfde OOS-periode.",
            "description": (
                "Experimentele BTC/ETH/cash-rotatie. Modelkwaliteit wordt uitsluitend uit "
                "eerdere OOS-uitkomsten afgeleid. Regime gebruikt alleen reeds gerealiseerde "
                "rendementen. Totale blootstelling is 0/25/50/75/100%; kapitaal gaat naar "
                "BTC en/of ETH op basis van kwaliteit- en confidence-gewogen signaalsterkte."
            ),
        },
    }
