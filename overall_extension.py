from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI

from v4_extension import dynamic_signal, market_regime

DATA_DIR = Path('/data')
WORKDIR = DATA_DIR / 'ultimate_run'
OUTPUT_DIR = WORKDIR / 'output'

HORIZON_WEIGHTS = {1: 0.10, 7: 0.20, 30: 0.30, 90: 0.40}


def _num(v, default=0.0):
    try:
        x = float(v)
        return default if np.isnan(x) or np.isinf(x) else x
    except Exception:
        return default


def overall_signals() -> list[dict]:
    latest = OUTPUT_DIR / 'latest_forecasts.csv'
    if not latest.exists():
        return []
    try:
        d = pd.read_csv(latest)
    except Exception:
        return []
    if d.empty or 'model' not in d.columns:
        return []
    d = d[d['model'].astype(str).str.upper() == 'META'].copy()
    out = []
    for coin in ('BTC', 'ETH'):
        g = d[d['coin'].astype(str).str.upper() == coin].copy()
        components = []
        weighted_prob = 0.0
        weighted_conf = 0.0
        used_weight = 0.0
        signed_votes = []
        for horizon, weight in HORIZON_WEIGHTS.items():
            z = g[pd.to_numeric(g['horizon'], errors='coerce') == horizon]
            if z.empty:
                continue
            r = z.iloc[-1]
            p = float(np.clip(_num(r.get('prob_up'), 0.5), 0.0, 1.0))
            pr = _num(r.get('pred_ret'), 0.0)
            conf = float(np.clip(_num(r.get('confidence_score'), 0.0), 0.0, 1.0))
            sig = dynamic_signal(coin, horizon, p, pr, conf)['label']
            effective_weight = weight * (0.55 + 0.45 * conf)
            weighted_prob += p * effective_weight
            weighted_conf += conf * weight
            used_weight += effective_weight
            vote = 1 if sig in ('BUY', 'STRONG BUY') else -1 if sig == 'SELL / REDUCE' else 0
            signed_votes.append(vote * weight)
            components.append({
                'horizon': horizon,
                'weight': weight,
                'probUp': round(p, 4),
                'predReturn': round(pr, 5),
                'confidence': round(conf, 4),
                'signal': sig,
            })
        if not components or used_weight <= 0:
            continue
        base_prob = weighted_prob / used_weight
        regime = market_regime(coin)
        regime_name = regime.get('regime', 'UNKNOWN')
        regime_conf = float(regime.get('confidence') or 0.0)
        regime_adj = 0.0
        if regime_name == 'BULLISH':
            regime_adj = 0.025 * regime_conf
        elif regime_name == 'BEARISH':
            regime_adj = -0.025 * regime_conf
        elif regime_name == 'HIGH VOLATILITY':
            # Pull extreme scores slightly toward neutral in very volatile markets.
            regime_adj = -0.015 * np.sign(base_prob - 0.5) * regime_conf
        combined_prob = float(np.clip(base_prob + regime_adj, 0.0, 1.0))
        avg_conf = float(np.clip(weighted_conf / sum(HORIZON_WEIGHTS[h['horizon']] for h in components), 0.0, 1.0))
        vote_sum = float(sum(signed_votes))
        agreement = float(np.clip(abs(vote_sum) / sum(HORIZON_WEIGHTS[h['horizon']] for h in components), 0.0, 1.0))
        overall_conf = float(np.clip(0.70 * avg_conf + 0.30 * agreement, 0.0, 1.0))

        if combined_prob >= 0.64 and overall_conf >= 0.60 and vote_sum > 0:
            label = 'STRONG BUY'
        elif combined_prob >= 0.57 and overall_conf >= 0.50 and vote_sum >= 0:
            label = 'BUY'
        elif combined_prob <= 0.36 and overall_conf >= 0.60 and vote_sum < 0:
            label = 'STRONG SELL'
        elif combined_prob <= 0.43 and overall_conf >= 0.50 and vote_sum <= 0:
            label = 'SELL / REDUCE'
        else:
            label = 'HOLD'

        score = round((combined_prob - 0.5) * 200, 1)  # -100 to +100
        out.append({
            'coin': coin,
            'signal': label,
            'score': score,
            'probUpComposite': round(combined_prob, 4),
            'confidence': round(overall_conf, 4),
            'agreement': round(agreement, 4),
            'regime': regime_name,
            'regimeAdjustment': round(float(regime_adj), 4),
            'components': components,
            'method': 'Weighted 1d/7d/30d/90d model signal; horizon weights 10/20/30/40%, confidence-adjusted, with a small capped market-regime adjustment.',
        })
    return out


def install(app: FastAPI) -> FastAPI:
    @app.get('/api/v1/overall')
    def overall_endpoint():
        return {
            'signals': overall_signals(),
            'disclaimer': 'Model signal for decision support; not a guarantee or personalized investment advice.'
        }

    for route in app.routes:
        if getattr(route, 'path', None) == '/api/v1/dashboard' and hasattr(route, 'dependant'):
            original = route.dependant.call
            def wrapped_dashboard(_orig=original):
                payload = _orig()
                if isinstance(payload, dict):
                    try:
                        payload['overallSignals'] = overall_signals()
                    except Exception:
                        payload['overallSignals'] = []
                return payload
            route.endpoint = wrapped_dashboard
            route.dependant.call = wrapped_dashboard
            break
    return app
