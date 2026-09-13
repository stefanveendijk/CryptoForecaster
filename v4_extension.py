from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI

DATA_DIR = Path('/data')
WORKDIR = DATA_DIR / 'ultimate_run'
OUTPUT_DIR = WORKDIR / 'output'
CACHE_DIR = WORKDIR / 'cache'
AUDIT_FILE = DATA_DIR / 'forecast_audit.csv'

FALLBACK_BUY = 0.57
FALLBACK_SELL = 0.43
MIN_CONFIDENCE = 0.50


def _num(v: Any):
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None


def _audit() -> pd.DataFrame:
    try:
        return pd.read_csv(AUDIT_FILE) if AUDIT_FILE.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def adaptive_thresholds(coin: str, horizon: int) -> dict:
    d = _audit()
    if d.empty:
        return {'buy': FALLBACK_BUY, 'sell': FALLBACK_SELL, 'source': 'fallback', 'n': 0, 'score': None}
    z = d[(d['coin'].astype(str).str.upper() == coin.upper()) & (pd.to_numeric(d['horizon'], errors='coerce') == int(horizon))].copy()
    if 'forecast_score' not in z.columns:
        return {'buy': FALLBACK_BUY, 'sell': FALLBACK_SELL, 'source': 'fallback', 'n': 0, 'score': None}
    z['forecast_score'] = pd.to_numeric(z['forecast_score'], errors='coerce')
    z = z[z['forecast_score'].notna()].tail(90)
    n = int(len(z))
    if n < 10:
        return {'buy': FALLBACK_BUY, 'sell': FALLBACK_SELL, 'source': 'fallback', 'n': n, 'score': float(z['forecast_score'].mean()) if n else None}
    score = float(z['forecast_score'].mean())
    # Better validated performance can use slightly less conservative thresholds;
    # weaker performance requires stronger evidence. Clamp prevents large behavior changes.
    adjustment = float(np.clip((65.0 - score) / 200.0, -0.04, 0.04))
    buy = float(np.clip(FALLBACK_BUY + adjustment, 0.53, 0.63))
    sell = float(np.clip(1.0 - buy, 0.37, 0.47))
    return {'buy': round(buy, 4), 'sell': round(sell, 4), 'source': 'empirical', 'n': n, 'score': round(score, 2)}


def dynamic_signal(coin: str, horizon: int, prob_up: float, pred_ret: float, confidence: float) -> dict:
    t = adaptive_thresholds(coin, horizon)
    p, r, c = float(prob_up), float(pred_ret), float(confidence or 0.0)
    if p >= max(0.64, t['buy'] + 0.05) and r > 0 and c >= 0.60:
        label = 'STRONG BUY'
    elif p >= t['buy'] and r > 0 and c >= MIN_CONFIDENCE:
        label = 'BUY'
    elif p <= t['sell'] and r < 0 and c >= MIN_CONFIDENCE:
        label = 'SELL / REDUCE'
    else:
        label = 'HOLD'
    return {'label': label, 'thresholds': t}


def _prices() -> pd.DataFrame:
    p = CACHE_DIR / 'crypto.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(p, parse_dates=['date']).sort_values('date')
        return d
    except Exception:
        return pd.DataFrame()


def market_regime(coin: str) -> dict:
    d = _prices()
    col = f'{coin.upper()}_close'
    if d.empty or col not in d.columns:
        return {'coin': coin.upper(), 'regime': 'UNKNOWN', 'confidence': 0.0}
    s = pd.to_numeric(d[col], errors='coerce').dropna().tail(120)
    if len(s) < 55:
        return {'coin': coin.upper(), 'regime': 'UNKNOWN', 'confidence': 0.0}
    ret30 = float(s.iloc[-1] / s.iloc[-31] - 1.0) if len(s) >= 31 else 0.0
    ma20 = float(s.tail(20).mean())
    ma50 = float(s.tail(50).mean())
    vol20 = float(s.pct_change().tail(20).std() * np.sqrt(365))
    trend = (ma20 / ma50 - 1.0) if ma50 else 0.0
    if ret30 > 0.08 and trend > 0.02:
        regime = 'BULLISH'
    elif ret30 < -0.08 and trend < -0.02:
        regime = 'BEARISH'
    elif vol20 >= 0.80:
        regime = 'HIGH VOLATILITY'
    else:
        regime = 'SIDEWAYS / NEUTRAL'
    strength = min(1.0, abs(ret30) / 0.20 + abs(trend) / 0.08)
    if regime == 'HIGH VOLATILITY':
        strength = min(1.0, vol20 / 1.2)
    return {
        'coin': coin.upper(), 'regime': regime, 'confidence': round(float(strength), 3),
        'return30d': round(ret30, 5), 'ma20VsMa50': round(trend, 5), 'annualizedVol20': round(vol20, 5)
    }


def backtest_summary() -> list[dict]:
    d = _audit()
    if d.empty or 'actual_ret' not in d.columns:
        return []
    d['actual_ret'] = pd.to_numeric(d['actual_ret'], errors='coerce')
    d['prob_up'] = pd.to_numeric(d['prob_up'], errors='coerce')
    d['pred_ret'] = pd.to_numeric(d['pred_ret'], errors='coerce')
    d['confidence'] = pd.to_numeric(d.get('confidence'), errors='coerce')
    d = d[d['actual_ret'].notna()].copy()
    out = []
    for (coin, horizon), g in d.groupby(['coin', 'horizon']):
        g = g.sort_values('forecast_date').tail(365).copy()
        returns = []
        active = 0
        wins = 0
        for _, r in g.iterrows():
            sig = dynamic_signal(str(coin), int(horizon), float(r.get('prob_up') or 0), float(r.get('pred_ret') or 0), float(r.get('confidence') or 0))['label']
            ar = float(r['actual_ret'])
            if sig in ('BUY', 'STRONG BUY'):
                sr = ar; active += 1; wins += int(sr > 0)
            elif sig == 'SELL / REDUCE':
                sr = -ar; active += 1; wins += int(sr > 0)
            else:
                sr = 0.0
            returns.append(sr)
        if not returns:
            continue
        curve = np.cumprod(1.0 + np.array(returns, dtype=float))
        peak = np.maximum.accumulate(curve)
        dd = curve / peak - 1.0
        out.append({
            'coin': str(coin), 'horizon': int(horizon), 'n': int(len(g)), 'activeSignals': int(active),
            'signalWinRate': round(float(wins / active), 4) if active else None,
            'strategyReturn': round(float(curve[-1] - 1.0), 5),
            'maxDrawdown': round(float(dd.min()), 5),
            'buyHoldComparableReturn': round(float(np.prod(1.0 + g['actual_ret'].to_numpy(dtype=float)) - 1.0), 5),
            'method': 'out-of-sample forecast audit; no fees/slippage; HOLD=0; SELL modeled short for comparison'
        })
    return out


def install(app: FastAPI) -> FastAPI:
    @app.get('/api/v1/v4')
    def v4_endpoint():
        latest = OUTPUT_DIR / 'latest_forecasts.csv'
        signals = []
        if latest.exists():
            try:
                d = pd.read_csv(latest)
                d = d[d['model'].astype(str).str.upper() == 'META']
                for _, r in d.iterrows():
                    coin = str(r.get('coin', '')).upper()
                    h = int(r.get('horizon', 0))
                    p = _num(r.get('prob_up')) or 0.0
                    pr = _num(r.get('pred_ret')) or 0.0
                    conf = _num(r.get('confidence_score')) or 0.0
                    s = dynamic_signal(coin, h, p, pr, conf)
                    signals.append({'coin': coin, 'horizon': h, 'signal': s['label'], 'thresholds': s['thresholds']})
            except Exception:
                pass
        return {
            'version': 'v4',
            'signals': signals,
            'regimes': [market_regime('BTC'), market_regime('ETH')],
            'backtest': backtest_summary(),
            'notes': ['Adaptive thresholds only activate after at least 10 matured forecasts per coin/horizon.', 'Backtest is descriptive and excludes fees/slippage.']
        }

    # Enrich dashboard but retain every existing field and route.
    for route in app.routes:
        if getattr(route, 'path', None) == '/api/v1/dashboard' and hasattr(route, 'dependant'):
            original = route.dependant.call
            def wrapped_dashboard(_orig=original):
                payload = _orig()
                forecasts = payload.get('forecasts', []) if isinstance(payload, dict) else []
                for f in forecasts:
                    try:
                        s = dynamic_signal(f.get('coin',''), int(f.get('horizon',0)), float(f.get('probUp') or 0), float(f.get('predReturn') or 0), float(f.get('confidenceScore') or 0))
                        f['tradeSignal'] = s['label']
                        f['signalThresholds'] = s['thresholds']
                    except Exception:
                        pass
                try:
                    payload['marketRegimes'] = [market_regime('BTC'), market_regime('ETH')]
                    payload['backtestSummary'] = backtest_summary()
                    payload['appVersion'] = 'v4'
                except Exception:
                    pass
                return payload
            route.endpoint = wrapped_dashboard
            route.dependant.call = wrapped_dashboard
            break
    return app
