from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEDGER_FILE = "forecast_ledger.csv"
RESULTS_FILE = "forecast_proof_results.csv"

LEDGER_COLUMNS = [
    "issued_at_utc", "forecast_date", "coin", "horizon", "model",
    "current_price", "prob_up", "pred_ret", "low80", "high80",
    "expected_price", "low80_price", "high80_price",
    "confidence_score", "ood_fraction", "expert_disagreement",
]


def _clean(v: Any) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def record_forecasts(output_dir: Path, latest: pd.DataFrame) -> dict[str, int]:
    """Append new META forecasts to an immutable issuance ledger.

    A key is forecast_date + coin + horizon + model. Re-running the model for the
    same model date never rewrites the original prediction.
    """
    path = output_dir / LEDGER_FILE
    existing = _read_csv(path)
    if latest is None or latest.empty:
        return {"existing": int(len(existing)), "added": 0}

    src = latest.copy()
    if "model" in src.columns:
        src = src[src["model"].astype(str).str.upper() == "META"].copy()
    if src.empty:
        return {"existing": int(len(existing)), "added": 0}

    if existing.empty:
        existing = pd.DataFrame(columns=LEDGER_COLUMNS)
    else:
        for c in LEDGER_COLUMNS:
            if c not in existing.columns:
                existing[c] = np.nan

    key_cols = ["forecast_date", "coin", "horizon", "model"]
    existing_keys = set()
    for _, r in existing.iterrows():
        try:
            existing_keys.add((
                str(pd.Timestamp(r["forecast_date"]).date()),
                str(r["coin"]).upper(),
                int(r["horizon"]),
                str(r["model"]).upper(),
            ))
        except Exception:
            continue

    issued_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for _, r in src.iterrows():
        try:
            forecast_date = str(pd.Timestamp(r.get("date")).date())
            coin = str(r.get("coin", "")).upper()
            horizon = int(r.get("horizon"))
            model = str(r.get("model", "META")).upper()
        except Exception:
            continue
        if not coin or horizon <= 0:
            continue
        key = (forecast_date, coin, horizon, model)
        if key in existing_keys:
            continue

        row = {
            "issued_at_utc": issued_at,
            "forecast_date": forecast_date,
            "coin": coin,
            "horizon": horizon,
            "model": model,
            "current_price": _clean(r.get("current_price")),
            "prob_up": _clean(r.get("prob_up")),
            "pred_ret": _clean(r.get("pred_ret")),
            "low80": _clean(r.get("low80")),
            "high80": _clean(r.get("high80")),
            "expected_price": _clean(r.get("expected_price")),
            "low80_price": _clean(r.get("low80_price")),
            "high80_price": _clean(r.get("high80_price")),
            "confidence_score": _clean(r.get("confidence_score")),
            "ood_fraction": _clean(r.get("ood_fraction")),
            "expert_disagreement": _clean(r.get("expert_disagreement")),
        }
        rows.append(row)
        existing_keys.add(key)

    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        combined = combined[LEDGER_COLUMNS]
        _atomic_csv(combined, path)

    return {"existing": int(len(existing)), "added": int(len(rows))}


def _normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    d = prices.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).set_index("date")
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].copy()
    try:
        d.index = d.index.tz_localize(None)
    except TypeError:
        try:
            d.index = d.index.tz_convert(None)
        except Exception:
            pass
    d.index = d.index.normalize()
    return d[~d.index.duplicated(keep="last")].sort_index()


def evaluate_ledger(output_dir: Path, prices: pd.DataFrame) -> pd.DataFrame:
    ledger = _read_csv(output_dir / LEDGER_FILE)
    px = _normalise_prices(prices)
    if ledger.empty or px.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    max_date = px.index.max()

    for _, r in ledger.iterrows():
        try:
            start_date = pd.Timestamp(r["forecast_date"]).normalize()
            horizon = int(r["horizon"])
            coin = str(r["coin"]).upper()
            target_date = start_date + pd.Timedelta(days=horizon)
        except Exception:
            continue

        price_col = f"{coin}_close"
        if price_col not in px.columns or target_date > max_date:
            continue

        start_price = _clean(r.get("current_price"))
        if start_price is None:
            if start_date in px.index:
                start_price = _clean(px.at[start_date, price_col])
        if start_price is None or start_price <= 0:
            continue

        eligible = px.loc[px.index >= target_date, price_col].dropna()
        if eligible.empty:
            continue
        actual_date = eligible.index[0]
        # Crypto trades every day; reject a large gap rather than silently using stale data.
        if (actual_date - target_date).days > 2:
            continue

        target_price = _clean(eligible.iloc[0])
        prob_up = _clean(r.get("prob_up"))
        pred_ret = _clean(r.get("pred_ret"))
        low80 = _clean(r.get("low80"))
        high80 = _clean(r.get("high80"))
        if target_price is None or prob_up is None or pred_ret is None:
            continue

        actual_ret = target_price / start_price - 1.0
        actual_up = 1 if actual_ret >= 0 else 0
        predicted_up = 1 if prob_up >= 0.5 else 0

        rows.append({
            "issued_at_utc": r.get("issued_at_utc"),
            "forecast_date": str(start_date.date()),
            "target_date": str(target_date.date()),
            "actual_date": str(actual_date.date()),
            "coin": coin,
            "horizon": horizon,
            "model": str(r.get("model", "META")).upper(),
            "start_price": start_price,
            "target_price": target_price,
            "prob_up": prob_up,
            "pred_ret": pred_ret,
            "actual_ret": actual_ret,
            "actual_up": actual_up,
            "predicted_up": predicted_up,
            "direction_correct": int(predicted_up == actual_up),
            "brier": (prob_up - actual_up) ** 2,
            "neutral_brier": 0.25,
            "abs_return_error": abs(pred_ret - actual_ret),
            "zero_return_abs_error": abs(actual_ret),
            "interval80_hit": (
                int(low80 <= actual_ret <= high80)
                if low80 is not None and high80 is not None else np.nan
            ),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["actual_date", "coin", "horizon"]).reset_index(drop=True)
        _atomic_csv(result, output_dir / RESULTS_FILE)
    return result


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _binom_two_sided_half(k: int, n: int) -> float | None:
    if n <= 0:
        return None
    if k >= n / 2:
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    else:
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _summary_row(d: pd.DataFrame, issued: int, coin: str, horizon: int) -> dict[str, Any]:
    n = int(len(d))
    pending = max(0, int(issued) - n)
    if n == 0:
        return {
            "coin": coin, "horizon": int(horizon), "issued": int(issued),
            "resolved": 0, "pending": pending, "sampleStatus": "collecting",
        }

    correct = int(d["direction_correct"].sum())
    acc = correct / n
    lo, hi = _wilson_interval(correct, n)
    brier = float(d["brier"].mean())
    baseline_brier = float(d["neutral_brier"].mean())
    mae = float(d["abs_return_error"].mean())
    baseline_mae = float(d["zero_return_abs_error"].mean())
    coverage = float(pd.to_numeric(d["interval80_hit"], errors="coerce").dropna().mean()) if d["interval80_hit"].notna().any() else None

    return {
        "coin": coin,
        "horizon": int(horizon),
        "issued": int(issued),
        "resolved": n,
        "pending": pending,
        "sampleStatus": "usable" if n >= 100 else ("early" if n >= 30 else "collecting"),
        "directionAccuracy": acc,
        "directionCorrect": correct,
        "directionCi95Low": lo,
        "directionCi95High": hi,
        "directionPValueVs50": _binom_two_sided_half(correct, n),
        "brier": brier,
        "neutralBrier": baseline_brier,
        "brierSkillVsNeutral": (1.0 - brier / baseline_brier) if baseline_brier > 0 else None,
        "maeReturn": mae,
        "zeroReturnMae": baseline_mae,
        "maeSkillVsZero": (1.0 - mae / baseline_mae) if baseline_mae > 0 else None,
        "coverage80": coverage,
        "firstForecast": str(d["forecast_date"].min()),
        "lastResolved": str(d["actual_date"].max()),
    }


def build_summary(output_dir: Path, prices: pd.DataFrame | None = None) -> dict[str, Any]:
    ledger = _read_csv(output_dir / LEDGER_FILE)
    if ledger.empty:
        return {
            "available": True,
            "started": False,
            "message": "De live bewijslaag start zodra de eerstvolgende modelrun een voorspelling vastlegt.",
            "summaries": [],
            "recent": [],
            "method": _method(),
        }

    if prices is not None and not prices.empty:
        results = evaluate_ledger(output_dir, prices)
    else:
        results = _read_csv(output_dir / RESULTS_FILE)

    summaries: list[dict[str, Any]] = []
    keys = (
        ledger[["coin", "horizon"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["coin", "horizon"])
    )
    for _, k in keys.iterrows():
        coin = str(k["coin"]).upper()
        horizon = int(k["horizon"])
        issued = int(((ledger["coin"].astype(str).str.upper() == coin) &
                      (pd.to_numeric(ledger["horizon"], errors="coerce") == horizon)).sum())
        if results.empty:
            d = pd.DataFrame()
        else:
            d = results[
                (results["coin"].astype(str).str.upper() == coin) &
                (pd.to_numeric(results["horizon"], errors="coerce") == horizon)
            ].copy()
        summaries.append(_summary_row(d, issued, coin, horizon))

    recent: list[dict[str, Any]] = []
    if not results.empty:
        tail = results.sort_values("actual_date").tail(20).iloc[::-1]
        for _, r in tail.iterrows():
            recent.append({
                "coin": str(r.get("coin", "")).upper(),
                "horizon": int(r.get("horizon", 0)),
                "forecastDate": str(r.get("forecast_date", "")),
                "targetDate": str(r.get("target_date", "")),
                "predReturn": _clean(r.get("pred_ret")),
                "actualReturn": _clean(r.get("actual_ret")),
                "probUp": _clean(r.get("prob_up")),
                "directionCorrect": bool(r.get("direction_correct")),
            })

    return {
        "available": True,
        "started": True,
        "issuedTotal": int(len(ledger)),
        "resolvedTotal": int(len(results)) if not results.empty else 0,
        "startedAt": str(ledger["issued_at_utc"].iloc[0]) if "issued_at_utc" in ledger else None,
        "summaries": summaries,
        "recent": recent,
        "method": _method(),
    }


def _method() -> dict[str, Any]:
    return {
        "ledger": (
            "Elke META-voorspelling wordt bij eerste uitgifte vastgelegd en bij een "
            "herberekening voor dezelfde modeldatum niet overschreven."
        ),
        "baseline": (
            "Richting wordt vergeleken met een neutrale 50/50-kans; rendement met "
            "een nul-rendementsvoorspelling."
        ),
        "statistics": (
            "De richting-hit-rate krijgt een Wilson 95%-interval en een verkennende "
            "exacte binomiale vergelijking met 50%. Vooral 30/90-daagse voorspellingen "
            "overlappen en zijn daardoor niet volledig onafhankelijk; de p-waarde mag "
            "dus niet als definitief bewijs worden gelezen."
        ),
        "sampleThresholds": {"early": 30, "usable": 100},
    }
