from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd


def clean(v):
    if v is None:
        return None
    try:
        n = float(v)
        return None if math.isnan(n) or math.isinf(n) else n
    except Exception:
        return None


def weights(v):
    if isinstance(v, dict):
        return v
    if not isinstance(v, str) or not v.strip():
        return {}
    try:
        x = json.loads(v)
        return {str(k): float(val) for k, val in x.items()}
    except Exception:
        return {}


def main() -> None:
    workdir = Path(sys.argv[1] if len(sys.argv) > 1 else ".pages_run")
    output = workdir / "output"
    site = Path(sys.argv[2] if len(sys.argv) > 2 else "pages_site")
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    latest_path = output / "latest_forecasts.csv"
    if not latest_path.exists():
        raise SystemExit(f"Missing model output: {latest_path}")

    latest = pd.read_csv(latest_path)
    latest = latest[latest["model"].astype(str).str.upper() == "META"].copy()
    forecasts = []
    for _, r in latest.iterrows():
        forecasts.append({
            "coin": str(r.get("coin", "")).upper(),
            "horizon": int(r.get("horizon", 0)),
            "date": str(r.get("date", "")),
            "probUp": clean(r.get("prob_up")) or 0,
            "predReturn": clean(r.get("pred_ret")) or 0,
            "low80": clean(r.get("low80")) or 0,
            "high80": clean(r.get("high80")) or 0,
            "currentPrice": clean(r.get("current_price")),
            "expectedPrice": clean(r.get("expected_price")),
            "low80Price": clean(r.get("low80_price")),
            "high80Price": clean(r.get("high80_price")),
            "confidenceScore": clean(r.get("confidence_score")),
            "oodFraction": clean(r.get("ood_fraction")),
            "expertDisagreement": clean(r.get("expert_disagreement")),
            "weights": weights(r.get("weights")),
        })

    metrics = []
    mp = output / "model_summary.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        m = m[m["expert"].astype(str).str.upper() == "META"]
        for _, r in m.iterrows():
            metrics.append({
                "coin": str(r.get("coin", "")).upper(),
                "horizon": int(r.get("horizon", 0)),
                "expert": "META",
                "auc": clean(r.get("auc")),
                "brier": clean(r.get("brier")),
                "accuracy": clean(r.get("accuracy")),
                "balancedAccuracy": clean(r.get("balanced_accuracy")),
                "maeReturn": clean(r.get("mae_return")),
            })

    events = []
    ep = output / "detected_events.csv"
    if ep.exists():
        e = pd.read_csv(ep).tail(50)
        for _, r in e.iloc[::-1].iterrows():
            events.append({
                "coin": str(r.get("coin", "")).upper(),
                "topic": str(r.get("topic", "")),
                "date": str(r.get("date", "")),
                "tone": clean(r.get("tone")),
                "sentiment": str(r.get("sentiment", "")),
                "articleCount": clean(r.get("article_count")),
                "eventZ90": clean(r.get("event_z90")),
            })

    updated = max((f["date"] for f in forecasts), default="")
    payload = {
        "updatedAt": updated,
        "source": "github-pages",
        "forecasts": forecasts,
        "metrics": metrics,
        "events": events,
    }
    (data_dir / "dashboard.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    hist_dir = data_dir / "history"
    hist_dir.mkdir(exist_ok=True)
    for coin in ("BTC", "ETH"):
        for horizon in (1, 7, 30, 90):
            p = output / f"{coin}_{horizon}d_META_oos.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p).tail(500)
            if "date" not in d.columns:
                d = d.rename(columns={d.columns[0]: "date"})
            points = []
            for _, r in d.iterrows():
                points.append({
                    "date": str(r.get("date", "")),
                    "probUp": clean(r.get("prob_up")) or 0,
                    "predReturn": clean(r.get("pred_ret")) or 0,
                    "actualReturn": clean(r.get("actual_ret")),
                })
            (hist_dir / f"{coin}_{horizon}.json").write_text(
                json.dumps({"coin": coin, "horizon": horizon, "points": points}, ensure_ascii=False),
                encoding="utf-8",
            )

    (site / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Exported {len(forecasts)} forecasts and {len(metrics)} metrics to {site}")


if __name__ == "__main__":
    main()
