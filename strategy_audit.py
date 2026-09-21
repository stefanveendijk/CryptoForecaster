from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import trading_strategy as ts

TRAIN_DAYS = 365
TEST_DAYS = 90

BUY_HALF_GRID = [0.04, 0.08, 0.12]
BUY_FULL_GRID = [0.16, 0.22, 0.28]
TRIM_GRID = [-0.02, 0.02, 0.06]
SELL_GRID = [-0.18, -0.12, -0.06]

_AUDIT_CACHE: dict[str, tuple[tuple, dict[str, Any]]] = {}


def _cache_key(output_dir: Path, coin: str) -> tuple:
    paths = [output_dir / f"{coin}_{h}d_META_oos.csv" for h in ts.HORIZONS]
    paths += [output_dir / "latest_forecasts.csv", output_dir / "price_history.csv"]
    key = []
    for p in paths:
        try:
            st = p.stat()
            key.append((p.name, st.st_mtime_ns, st.st_size))
        except OSError:
            key.append((p.name, None, None))
    return tuple(key)


def _target(score: float, current: float, p: dict[str, float]) -> float:
    if current <= 0.0:
        if score >= p["buyFull"]:
            return 1.0
        if score >= p["buyHalf"]:
            return 0.5
        return 0.0

    if current < 1.0:
        if score >= p["buyFull"]:
            return 1.0
        if score <= p["sell"]:
            return 0.0
        return 0.5

    if score <= p["sell"]:
        return 0.0
    if score < p["trim"]:
        return 0.5
    return 1.0


def _fixed_params() -> dict[str, float]:
    return {
        "buyHalf": float(ts.BUY_HALF),
        "buyFull": float(ts.BUY_FULL),
        "trim": float(ts.TRIM_FULL),
        "sell": float(ts.SELL),
    }


def _candidate_grid() -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for half in BUY_HALF_GRID:
        for full in BUY_FULL_GRID:
            for trim in TRIM_GRID:
                for sell in SELL_GRID:
                    if not (sell < trim < full):
                        continue
                    if not (half < full):
                        continue
                    out.append({
                        "buyHalf": float(half),
                        "buyFull": float(full),
                        "trim": float(trim),
                        "sell": float(sell),
                    })
    # Neem de actuele vaste strategie altijd expliciet mee.
    fixed = _fixed_params()
    if fixed not in out:
        out.append(fixed)
    return out


def _simulate(
    base: pd.DataFrame,
    params: dict[str, float],
    initial_position: float = 0.0,
    transaction_cost: float = ts.TRANSACTION_COST,
) -> tuple[pd.DataFrame, float]:
    if base.empty:
        return pd.DataFrame(), initial_position

    rows = []
    current = float(initial_position)
    for dt, row in base.iterrows():
        score = float(row["score"])
        bh = float(row["buyhold_return"])
        target = _target(score, current, params)
        turnover = abs(target - current)
        strategy_return = target * bh - turnover * transaction_cost
        rows.append((dt, score, bh, target, turnover, strategy_return))
        current = target

    result = pd.DataFrame(
        rows,
        columns=[
            "date", "score", "buyhold_return", "position",
            "turnover", "strategy_return",
        ],
    ).set_index("date")
    return result, current


def _objective(metrics: dict[str, Any]) -> float:
    if not metrics:
        return -1e9

    cagr = metrics.get("cagr")
    dd = metrics.get("maxDrawdown")
    sharpe = metrics.get("sharpe")
    trades = metrics.get("trades", 0)
    days = max(int(metrics.get("days", 0)), 1)
    if cagr is None or dd is None:
        return -1e9

    # Rendement blijft belangrijk, maar extreme drawdown en overtrading worden
    # bestraft. De +100%-benchmark wordt hier bewust NIET als optimalisatiedoel
    # gebruikt, om doelgedreven overfitting te voorkomen.
    years = max(days / 365.25, 0.25)
    trades_per_year = trades / years
    overtrade_penalty = max(0.0, trades_per_year - 24.0) * 0.004
    sharpe_term = 0.0 if sharpe is None else 0.12 * float(np.clip(sharpe, -2.0, 3.0))
    return (
        float(np.clip(cagr, -1.0, 3.0))
        + 0.55 * float(dd)
        + sharpe_term
        - overtrade_penalty
    )


def _choose_params(train: pd.DataFrame) -> tuple[dict[str, float], float]:
    best = _fixed_params()
    best_score = -1e9
    for p in _candidate_grid():
        sim, _ = _simulate(train, p)
        m = ts._metrics(sim)
        score = _objective(m)
        if score > best_score:
            best_score = score
            best = p
    return best, float(best_score)


def _params_key(p: dict[str, float]) -> str:
    return (
        f"half={p['buyHalf']:.2f}|full={p['buyFull']:.2f}|"
        f"trim={p['trim']:.2f}|sell={p['sell']:.2f}"
    )


def _walk_forward(base: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if len(base) < TRAIN_DAYS + TEST_DAYS:
        return pd.DataFrame(), []

    test_parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    current = 0.0
    start = TRAIN_DAYS

    while start < len(base):
        stop = min(start + TEST_DAYS, len(base))
        train = base.iloc[:start]
        test = base.iloc[start:stop]
        if len(test) < 20:
            break

        params, train_score = _choose_params(train)
        sim, current = _simulate(test, params, initial_position=current)
        test_parts.append(sim)

        fold_metrics = ts._metrics(sim)
        folds.append({
            "trainEnd": str(train.index[-1].date()),
            "testStart": str(test.index[0].date()),
            "testEnd": str(test.index[-1].date()),
            "params": params,
            "trainObjective": train_score,
            "testReturn": fold_metrics.get("totalReturn"),
            "testMaxDrawdown": fold_metrics.get("maxDrawdown"),
        })
        start = stop

    if not test_parts:
        return pd.DataFrame(), []
    return pd.concat(test_parts).sort_index(), folds


def _year_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for year, d in frame.groupby(frame.index.year):
        if len(d) < 60:
            continue
        m = ts._metrics(d)
        rows.append({
            "year": int(year),
            "days": int(len(d)),
            "return": m.get("totalReturn"),
            "buyHoldReturn": m.get("buyHoldTotalReturn"),
            "maxDrawdown": m.get("maxDrawdown"),
        })
    return rows


def _threshold_stability(folds: list[dict[str, Any]]) -> dict[str, Any]:
    if not folds:
        return {}
    keys = ["buyHalf", "buyFull", "trim", "sell"]
    values = {k: [float(f["params"][k]) for f in folds] for k in keys}
    counts = Counter(_params_key(f["params"]) for f in folds)
    common_key, common_n = counts.most_common(1)[0]
    return {
        "folds": len(folds),
        "mostCommonSet": common_key,
        "mostCommonShare": common_n / len(folds),
        "means": {k: float(np.mean(values[k])) for k in keys},
        "std": {k: float(np.std(values[k])) for k in keys},
    }


def _cost_stress(base: pd.DataFrame) -> list[dict[str, Any]]:
    fixed = _fixed_params()
    rows = []
    for cost in [0.001, 0.0025, 0.005]:
        sim, _ = _simulate(base, fixed, transaction_cost=cost)
        m = ts._metrics(sim)
        rows.append({
            "transactionCost": cost,
            "cagr": m.get("cagr"),
            "totalReturn": m.get("totalReturn"),
            "maxDrawdown": m.get("maxDrawdown"),
            "trades": m.get("trades"),
        })
    return rows


def build_audit(output_dir: Path, prices: pd.DataFrame, coin: str) -> dict[str, Any]:
    coin = coin.upper()
    if coin not in {"BTC", "ETH"}:
        raise ValueError("coin moet BTC of ETH zijn")

    cache_key = _cache_key(output_dir, coin)
    cached = _AUDIT_CACHE.get(coin)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    full_frame, _ = ts._backtest(output_dir, prices, coin)
    if full_frame.empty:
        result = {
            "coin": coin,
            "available": False,
            "reason": "Nog onvoldoende out-of-sample gegevens voor strategie-audit.",
        }
        _AUDIT_CACHE[coin] = (cache_key, result)
        return result

    base = full_frame[["score", "buyhold_return"]].dropna().copy()
    wf, folds = _walk_forward(base)
    if wf.empty or not folds:
        result = {
            "coin": coin,
            "available": False,
            "days": int(len(base)),
            "reason": (
                f"Minimaal {TRAIN_DAYS + TEST_DAYS} bruikbare OOS-dagen nodig; "
                f"nu {len(base)}."
            ),
        }
        _AUDIT_CACHE[coin] = (cache_key, result)
        return result

    # Vergelijk de vaste strategie exact over dezelfde ongeziene testperiode.
    holdout_base = base.loc[wf.index.min():wf.index.max()]
    fixed_holdout, _ = _simulate(holdout_base, _fixed_params())

    wf_metrics = ts._metrics(wf)
    fixed_metrics = ts._metrics(fixed_holdout)
    years = _year_rows(wf)
    positive_years = sum(1 for y in years if (y.get("return") or 0) > 0)

    stability = _threshold_stability(folds)
    latest_params = folds[-1]["params"] if folds else _fixed_params()

    wf_cagr = wf_metrics.get("cagr")
    bh_cagr = wf_metrics.get("buyHoldCagr")
    wf_dd = wf_metrics.get("maxDrawdown")

    checks = {
        "enoughWalkForwardFolds": len(folds) >= 4,
        "positiveWalkForwardCagr": wf_cagr is not None and wf_cagr > 0,
        "beatsBuyHoldCagr": (
            wf_cagr is not None and bh_cagr is not None and wf_cagr > bh_cagr
        ),
        "maxDrawdownBelow35Pct": wf_dd is not None and wf_dd >= -0.35,
        "annualTargetReached": wf_cagr is not None and wf_cagr >= ts.ANNUAL_TARGET,
        "majorityPositiveYears": bool(years) and positive_years > len(years) / 2,
    }

    result = {
        "coin": coin,
        "available": True,
        "method": {
            "trainDaysMinimum": TRAIN_DAYS,
            "testBlockDays": TEST_DAYS,
            "candidateSets": len(_candidate_grid()),
            "transactionCost": ts.TRANSACTION_COST,
            "annualTarget": ts.ANNUAL_TARGET,
            "description": (
                "Expanding walk-forward drempeltest. Elke testperiode gebruikt alleen "
                "drempels die op eerdere OOS-modelvoorspellingen zijn gekozen. De "
                "+100%-benchmark is geen optimalisatiedoel en beïnvloedt de selectie niet."
            ),
        },
        "walkForward": wf_metrics,
        "fixedStrategySamePeriod": fixed_metrics,
        "checks": checks,
        "foldCount": len(folds),
        "testStart": str(wf.index.min().date()),
        "testEnd": str(wf.index.max().date()),
        "calendarYears": years,
        "positiveYears": positive_years,
        "thresholdStability": stability,
        "latestSelectedThresholds": latest_params,
        "costStressFixedStrategy": _cost_stress(holdout_base),
        "folds": folds,
    }
    _AUDIT_CACHE[coin] = (cache_key, result)
    return result
