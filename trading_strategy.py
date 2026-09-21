from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

HORIZONS = [1, 7, 30, 90]

# Transparante, vooraf vastgelegde horizonweging.
# 30 dagen is de kern, 7 en 90 dagen geven respectievelijk timing en trend,
# 1 dag krijgt bewust weinig gewicht om overtrading te beperken.
HORIZON_WEIGHTS = {1: 0.10, 7: 0.25, 30: 0.40, 90: 0.25}

# Rendementsschalen voor normalisatie van de horizonvoorspellingen.
RETURN_SCALES = {1: 0.02, 7: 0.06, 30: 0.15, 90: 0.30}

TRANSACTION_COST = 0.001
ANNUAL_TARGET = 1.00  # +100% per jaar is een doelbenchmark, geen garantie.

# Hysterese: instappen vereist een sterker signaal dan vasthouden.
BUY_FULL = 0.22
BUY_HALF = 0.10
TRIM_FULL = 0.02
SELL = -0.10


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
    keep = [c for c in ["prob_up", "pred_ret", "actual_ret"] if c in d.columns]
    return d[keep]


def _component(prob_up: float, pred_ret: float, horizon: int) -> float:
    # Richting telt iets zwaarder dan de puntschatting van het rendement.
    direction = np.clip(2.0 * float(prob_up) - 1.0, -1.0, 1.0)
    ret_component = math.tanh(float(pred_ret) / RETURN_SCALES[horizon])
    return float(0.60 * direction + 0.40 * ret_component)


def _score_from_values(values: Dict[int, Dict[str, float]]) -> float | None:
    total = 0.0
    denom = 0.0
    for h, w in HORIZON_WEIGHTS.items():
        z = values.get(h)
        if not z:
            continue
        p = _clean(z.get("prob_up"))
        r = _clean(z.get("pred_ret"))
        if p is None or r is None:
            continue
        total += w * _component(p, r, h)
        denom += w
    if denom < 0.50:
        return None
    return float(np.clip(total / denom, -1.0, 1.0))


def _target_position(score: float, current: float) -> float:
    # Hysterese voorkomt dagelijks heen-en-weer handelen.
    if current <= 0.0:
        if score >= BUY_FULL:
            return 1.0
        if score >= BUY_HALF:
            return 0.5
        return 0.0

    if current < 1.0:
        if score >= BUY_FULL:
            return 1.0
        if score <= SELL:
            return 0.0
        return 0.5

    if score <= SELL:
        return 0.0
    if score < TRIM_FULL:
        return 0.5
    return 1.0


def _signal_name(previous: float, target: float) -> str:
    if target > previous:
        return "KOPEN"
    if target == 0.0 and previous > 0.0:
        return "VERKOPEN"
    if target < previous:
        return "AFBOUWEN"
    return "AANHOUDEN"


def _metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}

    r = frame["strategy_return"].dropna()
    bh = frame["buyhold_return"].dropna()
    if r.empty:
        return {}

    eq = (1.0 + r).cumprod()
    bheq = (1.0 + bh.reindex(r.index).fillna(0)).cumprod()
    years = max(len(r) / 365.25, 1 / 365.25)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    bh_cagr = float(bheq.iloc[-1] ** (1.0 / years) - 1.0) if bheq.iloc[-1] > 0 else -1.0

    dd = eq / eq.cummax() - 1.0
    sd = float(r.std())
    sharpe = float(r.mean() / sd * np.sqrt(365.25)) if sd > 0 else None

    trades = int((frame["position"].diff().abs().fillna(frame["position"]) > 0).sum())
    active = frame.loc[frame["position"] > 0, "strategy_return"]
    win_rate = float((active > 0).mean()) if len(active) else None

    return {
        "days": int(len(r)),
        "cagr": cagr,
        "buyHoldCagr": bh_cagr,
        "totalReturn": float(eq.iloc[-1] - 1.0),
        "buyHoldTotalReturn": float(bheq.iloc[-1] - 1.0),
        "maxDrawdown": float(dd.min()),
        "sharpe": sharpe,
        "trades": trades,
        "winRateActiveDays": win_rate,
        "averagePosition": float(frame.loc[r.index, "position"].mean()),
    }


def _backtest(output_dir: Path, prices: pd.DataFrame, coin: str) -> tuple[pd.DataFrame, dict]:
    price_col = f"{coin}_close"
    if price_col not in prices.columns:
        return pd.DataFrame(), {}

    horizon_frames = {}
    all_dates = set()
    for h in HORIZONS:
        d = _read_meta(output_dir, coin, h)
        if d.empty:
            continue
        horizon_frames[h] = d
        all_dates.update(d.index.tolist())

    if not all_dates:
        return pd.DataFrame(), {}

    score_rows = []
    for dt in sorted(all_dates):
        vals = {}
        for h, d in horizon_frames.items():
            if dt in d.index:
                row = d.loc[dt]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                vals[h] = {
                    "prob_up": row.get("prob_up"),
                    "pred_ret": row.get("pred_ret"),
                }
        score = _score_from_values(vals)
        if score is not None:
            score_rows.append((dt, score))

    if not score_rows:
        return pd.DataFrame(), {}

    scores = pd.Series(dict(score_rows), name="score").sort_index()
    price = pd.to_numeric(prices[price_col], errors="coerce").dropna().sort_index()
    idx = price.index.intersection(scores.index)
    if len(idx) < 30:
        return pd.DataFrame(), {}

    frame = pd.DataFrame(index=idx)
    frame["price"] = price.reindex(idx)
    frame["score"] = scores.reindex(idx)

    pos = []
    current = 0.0
    for score in frame["score"]:
        current = _target_position(float(score), current)
        pos.append(current)
    frame["position"] = pos

    # Signaal na dagclose t wordt toegepast op t -> t+1.
    daily_price = price.reindex(pd.date_range(price.index.min(), price.index.max(), freq="D")).ffill()
    next_ret = daily_price.pct_change().shift(-1)
    frame["buyhold_return"] = next_ret.reindex(frame.index)
    turnover = frame["position"].diff().abs().fillna(frame["position"])
    frame["strategy_return"] = frame["position"] * frame["buyhold_return"] - turnover * TRANSACTION_COST
    frame = frame.dropna(subset=["strategy_return", "buyhold_return"])

    return frame, _metrics(frame)


def _recent_metrics(frame: pd.DataFrame, days: int = 365) -> dict:
    if frame.empty:
        return {}
    end = frame.index.max()
    recent = frame.loc[frame.index > end - pd.Timedelta(days=days)].copy()
    return _metrics(recent)


def _load_latest(output_dir: Path, coin: str) -> Dict[int, Dict[str, float]]:
    p = output_dir / "latest_forecasts.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    if "model" in d.columns:
        d = d[d["model"].astype(str).str.upper().eq("META")]
    d = d[d["coin"].astype(str).str.upper().eq(coin.upper())]
    out = {}
    for _, r in d.iterrows():
        try:
            h = int(r.get("horizon"))
        except Exception:
            continue
        if h not in HORIZONS:
            continue
        out[h] = {
            "prob_up": _clean(r.get("prob_up")),
            "pred_ret": _clean(r.get("pred_ret")),
            "confidence": _clean(r.get("confidence_score")),
            "expected_price": _clean(r.get("expected_price")),
            "low80_price": _clean(r.get("low80_price")),
            "high80_price": _clean(r.get("high80_price")),
        }
    return out


def build_strategy(output_dir: Path, prices: pd.DataFrame, coin: str) -> dict:
    coin = coin.upper()
    if coin not in {"BTC", "ETH"}:
        raise ValueError("coin moet BTC of ETH zijn")

    frame, full = _backtest(output_dir, prices, coin)
    latest = _load_latest(output_dir, coin)
    score = _score_from_values(latest)

    previous = float(frame["position"].iloc[-1]) if not frame.empty else 0.0
    if score is None:
        target = previous
        signal = "GEEN SIGNAAL"
    else:
        target = _target_position(score, previous)
        signal = _signal_name(previous, target)

    recent = _recent_metrics(frame, 365)

    confidences = []
    for h, w in HORIZON_WEIGHTS.items():
        c = _clean(latest.get(h, {}).get("confidence"))
        if c is not None:
            confidences.append((w, c))
    confidence = (
        sum(w * c for w, c in confidences) / sum(w for w, _ in confidences)
        if confidences else None
    )

    horizon_view = {}
    for h in HORIZONS:
        z = latest.get(h, {})
        horizon_view[str(h)] = {
            "probUp": z.get("prob_up"),
            "predReturn": z.get("pred_ret"),
            "expectedPrice": z.get("expected_price"),
            "low80Price": z.get("low80_price"),
            "high80Price": z.get("high80_price"),
        }

    score100 = float(score * 100.0) if score is not None else None
    recent_return = recent.get("totalReturn")
    target_gap = recent_return - ANNUAL_TARGET if recent_return is not None else None

    reasons = []
    for h in [7, 30, 90, 1]:
        z = latest.get(h)
        if not z:
            continue
        p = z.get("prob_up")
        r = z.get("pred_ret")
        if p is not None and r is not None:
            reasons.append(f"{h}d: P(omhoog) {p*100:.1f}%, verwacht rendement {r*100:+.1f}%")

    return {
        "coin": coin,
        "signal": signal,
        "targetAllocation": target,
        "previousAllocation": previous,
        "signalScore": score100,
        "confidence": confidence,
        "annualTarget": ANNUAL_TARGET,
        "targetIsGoalNotGuarantee": True,
        "backtest": full,
        "last12Months": recent,
        "last12MonthsTargetGap": target_gap,
        "horizons": horizon_view,
        "reasons": reasons,
        "method": {
            "horizonWeights": {str(k): v for k, v in HORIZON_WEIGHTS.items()},
            "transactionCost": TRANSACTION_COST,
            "buyFullThreshold": BUY_FULL,
            "buyHalfThreshold": BUY_HALF,
            "trimThreshold": TRIM_FULL,
            "sellThreshold": SELL,
            "description": (
                "Long-only modelstrategie met hysterese. De 1/7/30/90-daagse META-"
                "voorspellingen worden gecombineerd. Backtest gebruikt uitsluitend "
                "out-of-sample modelvoorspellingen en past een signaal pas toe op de "
                "daaropvolgende dag."
            ),
        },
    }
