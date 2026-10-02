from __future__ import annotations

import math
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEDGER_FILE = "forecast_ledger.csv"
RESULTS_FILE = "forecast_proof_results.csv"

LEDGER_COLUMNS = [
    "issued_at_utc", "issue_date", "forecast_date", "coin", "horizon", "model",
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


def _parse_issued_at(value: Any = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        dt = pd.Timestamp(value).to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _issue_date_from_row(row: pd.Series) -> str:
    raw = row.get("issue_date")
    try:
        if raw is not None and not pd.isna(raw) and str(raw).strip():
            return str(pd.Timestamp(raw).date())
    except Exception:
        pass
    raw = row.get("issued_at_utc")
    try:
        if raw is not None and not pd.isna(raw) and str(raw).strip():
            return str(pd.Timestamp(raw).date())
    except Exception:
        pass
    try:
        return str(pd.Timestamp(row.get("forecast_date")).date())
    except Exception:
        return ""


def _migrate_ledger(existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    d = existing.copy()
    for c in LEDGER_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    d["issue_date"] = d.apply(_issue_date_from_row, axis=1)
    return d[LEDGER_COLUMNS]


def record_forecasts(
    output_dir: Path,
    latest: pd.DataFrame,
    issued_at_utc: Any = None,
) -> dict[str, int]:
    """Append one immutable META forecast per issuance day, coin and horizon.

    forecast_date remains the model's market-data date and is used for scoring.
    issue_date records the day the forecast was actually issued. Same-day reruns
    do not create duplicates. A late row is refused when its outcome could already
    have been known.
    """
    path = output_dir / LEDGER_FILE
    raw_existing = _read_csv(path)
    existing = _migrate_ledger(raw_existing)
    if latest is None or latest.empty:
        return {"existing": int(len(existing)), "added": 0, "skipped_late": 0}

    src = latest.copy()
    if "model" in src.columns:
        src = src[src["model"].astype(str).str.upper() == "META"].copy()
    if src.empty:
        return {"existing": int(len(existing)), "added": 0, "skipped_late": 0}

    issued_dt = _parse_issued_at(issued_at_utc)
    issued_at = issued_dt.isoformat()
    issue_date = str(issued_dt.date())

    existing_keys: set[tuple[str, str, int, str]] = set()
    for _, r in existing.iterrows():
        try:
            existing_keys.add((
                _issue_date_from_row(r),
                str(r["coin"]).upper(),
                int(r["horizon"]),
                str(r["model"]).upper(),
            ))
        except Exception:
            continue

    rows: list[dict[str, Any]] = []
    skipped_late = 0
    for _, r in src.iterrows():
        try:
            forecast_date_obj = pd.Timestamp(r.get("date")).date()
            forecast_date = str(forecast_date_obj)
            coin = str(r.get("coin", "")).upper()
            horizon = int(r.get("horizon"))
            model = str(r.get("model", "META")).upper()
        except Exception:
            continue
        if not coin or horizon <= 0:
            continue

        target_date = forecast_date_obj + timedelta(days=horizon)
        outcome_cutoff = datetime.combine(
            target_date + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        if issued_dt >= outcome_cutoff:
            skipped_late += 1
            continue

        key = (issue_date, coin, horizon, model)
        if key in existing_keys:
            continue

        rows.append({
            "issued_at_utc": issued_at,
            "issue_date": issue_date,
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
        })
        existing_keys.add(key)

    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        _atomic_csv(combined[LEDGER_COLUMNS], path)
    elif not existing.empty and "issue_date" not in raw_existing.columns:
        _atomic_csv(existing, path)

    return {
        "existing": int(len(existing)),
        "added": int(len(rows)),
        "skipped_late": int(skipped_late),
    }


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
    ledger = _migrate_ledger(_read_csv(output_dir / LEDGER_FILE))
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
        if start_price is None and start_date in px.index:
            start_price = _clean(px.at[start_date, price_col])
        if start_price is None or start_price <= 0:
            continue

        eligible = px.loc[px.index >= target_date, price_col].dropna()
        if eligible.empty:
            continue
        actual_date = eligible.index[0]
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
            "issue_date": _issue_date_from_row(r),
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
        result = result.sort_values(
            ["actual_date", "coin", "horizon", "issue_date"]
        ).reset_index(drop=True)
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


def _summary_row(resolved: pd.DataFrame, issued: int, coin: str, horizon: int) -> dict[str, Any]:
    resolved_n = int(len(resolved))
    pending = max(0, int(issued) - resolved_n)
    if resolved_n == 0:
        return {
            "coin": coin, "horizon": int(horizon), "issued": int(issued),
            "resolved": 0, "pending": pending, "metricSample": 0,
            "duplicateResolvedExcluded": 0, "sampleStatus": "collecting",
        }

    metric_df = (
        resolved.sort_values(["forecast_date", "issue_date"])
        .drop_duplicates(
            subset=["forecast_date", "coin", "horizon", "model"], keep="first"
        )
        .copy()
    )
    n = int(len(metric_df))
    duplicate_excluded = resolved_n - n
    correct = int(metric_df["direction_correct"].sum())
    acc = correct / n
    lo, hi = _wilson_interval(correct, n)
    brier = float(metric_df["brier"].mean())
    baseline_brier = float(metric_df["neutral_brier"].mean())
    mae = float(metric_df["abs_return_error"].mean())
    baseline_mae = float(metric_df["zero_return_abs_error"].mean())
    coverage = (
        float(pd.to_numeric(metric_df["interval80_hit"], errors="coerce").dropna().mean())
        if metric_df["interval80_hit"].notna().any() else None
    )

    return {
        "coin": coin,
        "horizon": int(horizon),
        "issued": int(issued),
        "resolved": resolved_n,
        "pending": pending,
        "metricSample": n,
        "duplicateResolvedExcluded": int(duplicate_excluded),
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
        "firstForecast": str(metric_df["forecast_date"].min()),
        "lastResolved": str(resolved["actual_date"].max()),
    }


def build_summary(output_dir: Path, prices: pd.DataFrame | None = None) -> dict[str, Any]:
    ledger = _migrate_ledger(_read_csv(output_dir / LEDGER_FILE))
    if ledger.empty:
        return {
            "available": True, "started": False,
            "message": "De live bewijslaag start zodra de eerstvolgende modelrun een voorspelling vastlegt.",
            "summaries": [], "recent": [], "method": _method(),
        }

    if prices is not None and not prices.empty:
        results = evaluate_ledger(output_dir, prices)
    else:
        results = _read_csv(output_dir / RESULTS_FILE)

    summaries: list[dict[str, Any]] = []
    keys = (
        ledger[["coin", "horizon"]].dropna().drop_duplicates().sort_values(["coin", "horizon"])
    )
    for _, k in keys.iterrows():
        coin = str(k["coin"]).upper()
        horizon = int(k["horizon"])
        ld = ledger[
            (ledger["coin"].astype(str).str.upper() == coin)
            & (pd.to_numeric(ledger["horizon"], errors="coerce") == horizon)
        ].copy()
        if results.empty:
            d = pd.DataFrame()
        else:
            d = results[
                (results["coin"].astype(str).str.upper() == coin)
                & (pd.to_numeric(results["horizon"], errors="coerce") == horizon)
            ].copy()
        row = _summary_row(d, len(ld), coin, horizon)
        issue_dates = pd.to_datetime(ld["issue_date"], errors="coerce").dropna()
        row["firstIssueDate"] = str(issue_dates.min().date()) if not issue_dates.empty else None
        row["lastIssueDate"] = str(issue_dates.max().date()) if not issue_dates.empty else None
        summaries.append(row)

    recent: list[dict[str, Any]] = []
    if not results.empty:
        tail = results.sort_values(["actual_date", "issue_date"]).tail(20).iloc[::-1]
        for _, r in tail.iterrows():
            recent.append({
                "coin": str(r.get("coin", "")).upper(),
                "horizon": int(r.get("horizon", 0)),
                "issueDate": str(r.get("issue_date", "")),
                "forecastDate": str(r.get("forecast_date", "")),
                "targetDate": str(r.get("target_date", "")),
                "predReturn": _clean(r.get("pred_ret")),
                "actualReturn": _clean(r.get("actual_ret")),
                "probUp": _clean(r.get("prob_up")),
                "directionCorrect": bool(r.get("direction_correct")),
            })

    all_issue_dates = pd.to_datetime(ledger["issue_date"], errors="coerce").dropna()
    return {
        "available": True,
        "started": True,
        "issuedTotal": int(len(ledger)),
        "resolvedTotal": int(len(results)) if not results.empty else 0,
        "startedAt": str(ledger["issued_at_utc"].iloc[0]) if "issued_at_utc" in ledger else None,
        "latestIssueDate": str(all_issue_dates.max().date()) if not all_issue_dates.empty else None,
        "summaries": summaries,
        "recent": recent,
        "method": _method(),
    }


def _method() -> dict[str, Any]:
    return {
        "ledger": (
            "Per uitgiftedag wordt voor iedere coin en horizon één META-voorspelling "
            "onveranderlijk vastgelegd. De oorspronkelijke modeldatum blijft apart "
            "bewaard voor de beoordeling."
        ),
        "independence": (
            "Als twee uitgiftedagen dezelfde modeldatum gebruiken, worden beide uitgiftes "
            "geregistreerd maar telt die modeldatum slechts één keer mee in de statistische vaardigheidsmeting."
        ),
        "baseline": (
            "Richting wordt vergeleken met een neutrale 50/50-kans; rendement met een nul-rendementsvoorspelling."
        ),
        "statistics": (
            "De richting-hit-rate krijgt een Wilson 95%-interval en een verkennende exacte "
            "binomiale vergelijking met 50%. Vooral 30/90-daagse voorspellingen overlappen "
            "en zijn daardoor niet volledig onafhankelijk; de p-waarde is dus niet definitief bewijs."
        ),
        "sampleThresholds": {"early": 30, "usable": 100},
    }
