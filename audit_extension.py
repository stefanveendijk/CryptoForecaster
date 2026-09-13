from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI

DATA_DIR = Path("/data")
WORKDIR = DATA_DIR / "ultimate_run"
OUTPUT_DIR = WORKDIR / "output"
CACHE_DIR = WORKDIR / "cache"
AUDIT_FILE = DATA_DIR / "forecast_audit.csv"

HORIZON_ERROR_SCALE = {1: 0.03, 7: 0.08, 30: 0.18, 90: 0.30}


def _num(v: Any):
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None


def _signal(prob_up: float, pred_ret: float, confidence: float) -> str:
    if prob_up >= 0.64 and pred_ret > 0 and confidence >= 0.60:
        return "STRONG BUY"
    if prob_up >= 0.57 and pred_ret > 0 and confidence >= 0.50:
        return "BUY"
    if prob_up <= 0.43 and pred_ret < 0 and confidence >= 0.50:
        return "SELL / REDUCE"
    return "HOLD"


def _load_audit() -> pd.DataFrame:
    if not AUDIT_FILE.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(AUDIT_FILE)
        return d
    except Exception:
        return pd.DataFrame()


def _save_audit(d: pd.DataFrame) -> None:
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(AUDIT_FILE, index=False)


def archive_latest_forecasts() -> None:
    path = OUTPUT_DIR / "latest_forecasts.csv"
    if not path.exists():
        return
    try:
        d = pd.read_csv(path)
    except Exception:
        return
    if d.empty or "model" not in d.columns:
        return
    d = d[d["model"].astype(str).str.upper() == "META"].copy()
    if d.empty:
        return

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for _, r in d.iterrows():
        coin = str(r.get("coin", "")).upper()
        horizon = int(r.get("horizon", 0))
        forecast_date = str(r.get("date", ""))[:10]
        p = _num(r.get("prob_up"))
        pr = _num(r.get("pred_ret"))
        cp = _num(r.get("current_price"))
        ep = _num(r.get("expected_price"))
        conf = _num(r.get("confidence_score"))
        if not coin or not horizon or not forecast_date or p is None or pr is None or cp is None:
            continue
        conf = conf if conf is not None else 0.0
        rows.append({
            "coin": coin,
            "horizon": horizon,
            "forecast_date": forecast_date,
            "created_at": now,
            "prob_up": p,
            "pred_ret": pr,
            "current_price": cp,
            "expected_price": ep,
            "confidence": conf,
            "signal": _signal(p, pr, conf),
            "actual_date": None,
            "actual_price": None,
            "actual_ret": None,
            "actual_up": None,
            "direction_correct": None,
            "brier": None,
            "abs_return_error": None,
            "forecast_score": None,
        })

    if not rows:
        return
    new = pd.DataFrame(rows)
    old = _load_audit()
    if old.empty:
        combined = new
    else:
        combined = pd.concat([old, new], ignore_index=True)
    combined = combined.sort_values(["forecast_date", "coin", "horizon", "created_at"])
    combined = combined.drop_duplicates(["forecast_date", "coin", "horizon"], keep="last")
    _save_audit(combined)


def _load_prices() -> pd.DataFrame:
    path = CACHE_DIR / "crypto.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d
    except Exception:
        return pd.DataFrame()


def evaluate_matured() -> None:
    audit = _load_audit()
    prices = _load_prices()
    if audit.empty or prices.empty:
        return

    changed = False
    for i, r in audit.iterrows():
        if pd.notna(r.get("actual_ret")):
            continue
        coin = str(r.get("coin", "")).upper()
        horizon = int(r.get("horizon", 0))
        try:
            fd = pd.Timestamp(r.get("forecast_date")).normalize()
        except Exception:
            continue
        target = fd + pd.Timedelta(days=horizon)
        price_col = f"{coin}_close"
        if price_col not in prices.columns:
            continue
        eligible = prices.loc[prices.index >= target, price_col].dropna()
        if eligible.empty:
            continue
        actual_date = eligible.index[0]
        actual_price = float(eligible.iloc[0])
        current_price = _num(r.get("current_price"))
        prob_up = _num(r.get("prob_up"))
        pred_ret = _num(r.get("pred_ret"))
        if current_price is None or current_price <= 0 or prob_up is None or pred_ret is None:
            continue

        actual_ret = actual_price / current_price - 1.0
        actual_up = 1 if actual_ret > 0 else 0
        direction_correct = 1 if ((prob_up >= 0.5) == bool(actual_up)) else 0
        brier = (prob_up - actual_up) ** 2
        abs_error = abs(pred_ret - actual_ret)
        scale = HORIZON_ERROR_SCALE.get(horizon, min(0.35, 0.03 * math.sqrt(max(1, horizon))))
        return_component = math.exp(-abs_error / max(scale, 1e-6))
        score = 100.0 * (0.40 * direction_correct + 0.35 * (1.0 - brier) + 0.25 * return_component)
        score = float(np.clip(score, 0.0, 100.0))

        audit.at[i, "actual_date"] = str(actual_date.date())
        audit.at[i, "actual_price"] = actual_price
        audit.at[i, "actual_ret"] = actual_ret
        audit.at[i, "actual_up"] = actual_up
        audit.at[i, "direction_correct"] = direction_correct
        audit.at[i, "brier"] = brier
        audit.at[i, "abs_return_error"] = abs_error
        audit.at[i, "forecast_score"] = score
        changed = True

    if changed:
        _save_audit(audit)


def audit_summary() -> list[dict]:
    archive_latest_forecasts()
    evaluate_matured()
    d = _load_audit()
    if d.empty:
        return []
    matured = d[pd.to_numeric(d.get("forecast_score"), errors="coerce").notna()].copy()
    if matured.empty:
        return []

    matured["forecast_date"] = pd.to_datetime(matured["forecast_date"], errors="coerce")
    now = pd.Timestamp.utcnow().tz_localize(None)
    out = []
    for (coin, horizon), g in matured.groupby(["coin", "horizon"]):
        for window in [30, 90, 365]:
            z = g[g["forecast_date"] >= now - pd.Timedelta(days=window)].copy()
            if z.empty:
                continue
            out.append({
                "coin": str(coin),
                "horizon": int(horizon),
                "windowDays": int(window),
                "n": int(len(z)),
                "directionAccuracy": float(pd.to_numeric(z["direction_correct"], errors="coerce").mean()),
                "brier": float(pd.to_numeric(z["brier"], errors="coerce").mean()),
                "maeReturn": float(pd.to_numeric(z["abs_return_error"], errors="coerce").mean()),
                "forecastScore": float(pd.to_numeric(z["forecast_score"], errors="coerce").mean()),
            })
    return out


def recent_audit(limit: int = 100) -> list[dict]:
    archive_latest_forecasts()
    evaluate_matured()
    d = _load_audit()
    if d.empty:
        return []
    d = d.sort_values(["forecast_date", "coin", "horizon"], ascending=[False, True, True]).head(limit)
    out = []
    for _, r in d.iterrows():
        item = {}
        for k, v in r.to_dict().items():
            if pd.isna(v):
                item[k] = None
            elif isinstance(v, (np.integer,)):
                item[k] = int(v)
            elif isinstance(v, (np.floating,)):
                item[k] = float(v)
            else:
                item[k] = v
        out.append(item)
    return out


def install(app: FastAPI) -> FastAPI:
    @app.get("/api/v1/audit")
    def audit_endpoint(limit: int = 100):
        return {
            "summary": audit_summary(),
            "records": recent_audit(max(1, min(int(limit), 500))),
            "scoreMethod": {
                "directionWeight": 0.40,
                "probabilityWeight": 0.35,
                "returnAccuracyWeight": 0.25,
                "note": "Forecast Score 0-100 combines direction, Brier probability quality and return error. It is descriptive, not a guarantee of future trading results.",
            },
        }

    # Enrich the existing dashboard response without changing cloud_server.py.
    for route in app.routes:
        if getattr(route, "path", None) == "/api/v1/dashboard" and hasattr(route, "dependant"):
            original = route.dependant.call
            def wrapped_dashboard(_orig=original):
                payload = _orig()
                try:
                    payload["auditSummary"] = audit_summary()
                except Exception:
                    payload["auditSummary"] = []
                return payload
            route.endpoint = wrapped_dashboard
            route.dependant.call = wrapped_dashboard
            break
    return app
