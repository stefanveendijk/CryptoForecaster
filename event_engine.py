"""
BTC/ETH Predictor v3 — koers + algemeen nieuws + eventcategorieën
==================================================================

Deze runner bouwt voort op v2 en voegt een expliciete eventlaag toe.

Drie modellen worden eerlijk met dezelfde out-of-sample dagen vergeleken:
A) PRICE        : alleen koers/volume/technische kenmerken
B) NEWS         : PRICE + generiek nieuwsvolume/sentiment
C) EVENT        : NEWS + categorie-specifieke eventfeatures

Eventcategorieën:
- regulation_sec
- etf_flows
- security_hacks
- bankruptcies_exchange
- stablecoins
- fed_rates_macro
- institutional_adoption
- geopolitics
- protocol_technology

Daarnaast:
- detectie van echte nieuwsuitbarstingen via 90-daagse z-score
- cooldown om hetzelfde incident niet meerdere keren te tellen
- event study op 1/3/7/30 dagen
- matched-control vergelijking
- FDR-correctie voor multiple testing
- richtingstest (nieuws -> rendement versus rendement -> nieuws)
- alleen vooraf beschikbare nieuwsfeatures gaan het voorspelmodel in

Gebruik:
    python crypto_predictor_event_v3.py

Optioneel:
    python crypto_predictor_event_v3.py --force-news-refresh
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import news_core as core


# ---------------------------------------------------------------------------
# Event-taxonomie
# ---------------------------------------------------------------------------

EVENT_TOPICS = {
    "regulation_sec": (
        '(regulation OR regulator OR regulated OR SEC OR lawsuit OR ban '
        'OR legislation OR MiCA OR enforcement OR approval)'
    ),
    "etf_flows": (
        '("spot bitcoin ETF" OR "spot ethereum ETF" OR "ETF inflow" OR '
        '"ETF outflow" OR BlackRock OR Fidelity OR Grayscale)'
    ),
    "security_hacks": (
        '(hack OR hacked OR exploit OR breach OR theft OR stolen OR '
        'cyberattack OR vulnerability)'
    ),
    "bankruptcies_exchange": (
        '(bankruptcy OR bankrupt OR insolvency OR liquidation OR '
        '"exchange collapse" OR FTX OR Celsius OR Genesis OR MtGox)'
    ),
    "stablecoins": (
        '(stablecoin OR Tether OR USDT OR USDC OR depeg OR depegging OR '
        'reserve OR redemption)'
    ),
    "fed_rates_macro": (
        '("Federal Reserve" OR FOMC OR "interest rate" OR rates OR inflation '
        'OR CPI OR jobs OR payrolls OR recession)'
    ),
    "institutional_adoption": (
        '("institutional adoption" OR institutional OR treasury OR '
        '"balance sheet" OR custody OR bank OR corporation OR corporate)'
    ),
    "geopolitics": (
        '(war OR conflict OR sanctions OR tariff OR geopolitical OR invasion '
        'OR election OR capital-controls)'
    ),
    "protocol_technology": (
        '(upgrade OR fork OR halving OR merge OR staking OR protocol OR '
        'validator OR scaling OR layer-2)'
    ),
}

# Core v2 gebruikt deze dict al voor categorie-volume.
core.TOPIC_QUERIES = EVENT_TOPICS

EVENT_Z_THRESHOLD = 2.0
MIN_EVENT_ARTICLES = 5
EVENT_COOLDOWN_DAYS = 3
MODEL_NEWS_LAG_DAYS = 1

# Generieke nieuwsfeatures die model B mag zien.
GENERIC_SUFFIXES = [
    "article_count", "tone", "negative_count", "positive_count",
    "volume_share", "negative_share", "positive_share", "sentiment_balance",
    "tone_ma3", "tone_ma7", "tone_ma30",
    "volume_ma3", "volume_ma7", "volume_ma30",
    "neg_ma3", "neg_ma7", "neg_ma30",
    "pos_ma3", "pos_ma7", "pos_ma30",
    "volume_z30", "volume_z90", "tone_z30", "tone_z90",
    "tone_change1", "tone_change3", "tone_change7",
    "volume_change1", "volume_change3", "volume_change7",
]


def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n.v.t."
    return f"{100*x:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# Categorie-specifieke tone-data
# ---------------------------------------------------------------------------

def load_event_tone_cache(
    cache_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force: bool = False,
) -> pd.DataFrame:
    """
    Haalt per munt en categorie de gemiddelde GDELT-tone op.
    De volume-data wordt al door core.load_or_update_news opgehaald.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for coin in core.COINS:
        path = cache_dir / f"gdelt_{coin.lower()}_event_tone.csv"
        existing = pd.DataFrame()

        if path.exists() and not force:
            existing = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
            existing.index = pd.to_datetime(existing.index).tz_localize(None).normalize()

        ranges = []
        if force or existing.empty:
            ranges = [(start, end)]
        else:
            if existing.index.min() > start:
                ranges.append((start, existing.index.min() - pd.Timedelta(days=1)))
            if existing.index.max() < end:
                ranges.append((existing.index.max() + pd.Timedelta(days=1), end))

        pieces = [existing] if not existing.empty else []

        for a, b in ranges:
            if a > b:
                continue
            idx = pd.date_range(a, b, freq="D")
            frame = pd.DataFrame(index=idx)

            print(f"  Categorie-tone {coin}: {a.date()} t/m {b.date()}")
            for topic, query in EVENT_TOPICS.items():
                q = f"{core.COIN_QUERY[coin]} {query}"
                tf = core.fetch_gdelt_timeline(q, "timelinetone", a, b)
                tone = core._series_from_timeline(tf, ["tone", core.COIN_QUERY[coin]])
                frame[f"{coin}_news_{topic}_tone"] = tone.reindex(idx)
                time.sleep(0.15)

            pieces.append(frame)

        combined = pd.concat(pieces).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.index.name = "date"
        combined.to_csv(path)
        all_frames.append(combined)

    return pd.concat(all_frames, axis=1).sort_index()


def add_event_features(
    model_df: pd.DataFrame,
    raw_news: pd.DataFrame,
    raw_event_tone: pd.DataFrame,
) -> pd.DataFrame:
    """
    model_df bevat al de 1-dag vertraagde v2-newsfeatures.
    We bouwen eventfeatures uit RAW data en vertragen pas op het eind 1 dag.
    """
    x = model_df.copy()
    idx = x.index

    raw = raw_news.join(raw_event_tone, how="outer").reindex(idx)

    for coin in core.COINS:
        for topic in EVENT_TOPICS:
            count_col = f"{coin}_news_{topic}_count"
            tone_col = f"{coin}_news_{topic}_tone"

            count = pd.to_numeric(raw.get(count_col), errors="coerce")
            tone = pd.to_numeric(raw.get(tone_col), errors="coerce")

            if count is None:
                continue

            # Geen artikelen in een categorie = neutrale tone, geen ontbrekende feature.
            # Dit voorkomt dat categorieën die pas later relevant werden (bv. spot ETF's)
            # de hele modelhistorie terugbrengen tot alleen recente jaren.
            tone = tone.where(count.fillna(0) > 0, 0.0).fillna(0.0)

            logc = np.log1p(count.fillna(0))
            mean90 = logc.rolling(90, min_periods=45).mean()
            std90 = logc.rolling(90, min_periods=45).std().replace(0, np.nan)
            z90 = ((logc - mean90) / std90).fillna(0.0)

            mean30 = logc.rolling(30, min_periods=15).mean()
            std30 = logc.rolling(30, min_periods=15).std().replace(0, np.nan)
            z30 = ((logc - mean30) / std30).fillna(0.0)

            tone_ma30 = tone.rolling(30, min_periods=10).mean()
            tone_sd30 = tone.rolling(30, min_periods=10).std().replace(0, np.nan)
            tone_z30 = ((tone - tone_ma30) / tone_sd30).fillna(0.0)

            share_col = f"{coin}_news_{topic}_share"
            if share_col in model_df:
                # Deze core-share is al 1 dag vertraagd. Voor eventfeatures hieronder
                # gebruiken we daarom niet opnieuw die kolom maar de raw count.
                pass

            # Event intensity combineert nieuwsuitbarsting en absolute emotionele lading.
            emotional = (tone.abs() / 5.0).clip(0, 3).fillna(0)
            intensity = z90.clip(lower=0).fillna(0) * (1.0 + 0.25 * emotional)

            derived = pd.DataFrame(index=idx)
            derived[f"{coin}_news_{topic}_event_z30"] = z30
            derived[f"{coin}_news_{topic}_event_z90"] = z90
            derived[f"{coin}_news_{topic}_event_tone"] = tone
            derived[f"{coin}_news_{topic}_event_tone_z30"] = tone_z30
            derived[f"{coin}_news_{topic}_event_intensity"] = intensity
            derived[f"{coin}_news_{topic}_event_negative"] = (
                ((z90 >= EVENT_Z_THRESHOLD) & (count >= MIN_EVENT_ARTICLES) & (tone < -2))
                .astype(float)
            )
            derived[f"{coin}_news_{topic}_event_positive"] = (
                ((z90 >= EVENT_Z_THRESHOLD) & (count >= MIN_EVENT_ARTICLES) & (tone > 2))
                .astype(float)
            )

            # Anti-leakage: eventinformatie van dag t is pas feature op dag t+1.
            derived = derived.shift(MODEL_NEWS_LAG_DAYS)
            x = x.join(derived, how="left")

    return x.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Eventdetectie en event study
# ---------------------------------------------------------------------------

def detect_events(
    price_df: pd.DataFrame,
    raw_news: pd.DataFrame,
    raw_event_tone: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for coin in core.COINS:
        for topic in EVENT_TOPICS:
            ccol = f"{coin}_news_{topic}_count"
            tcol = f"{coin}_news_{topic}_tone"
            if ccol not in raw_news.columns:
                continue

            count = pd.to_numeric(raw_news[ccol], errors="coerce").reindex(price_df.index)
            tone = (
                pd.to_numeric(raw_event_tone[tcol], errors="coerce").reindex(price_df.index)
                if tcol in raw_event_tone.columns else pd.Series(index=price_df.index, dtype=float)
            )

            logc = np.log1p(count.fillna(0))
            z = (
                (logc - logc.rolling(90, min_periods=45).mean())
                / logc.rolling(90, min_periods=45).std().replace(0, np.nan)
            )

            candidates = pd.DataFrame({
                "count": count,
                "tone": tone,
                "z": z,
            }).dropna(subset=["z"])

            candidates = candidates[
                (candidates["z"] >= EVENT_Z_THRESHOLD)
                & (candidates["count"] >= MIN_EVENT_ARTICLES)
            ]

            # Cooldown: binnen 3 dagen blijft alleen de sterkste nieuwsuitbarsting over.
            selected = []
            last_cluster = []
            for dt, r in candidates.iterrows():
                if not last_cluster:
                    last_cluster = [(dt, r)]
                    continue

                if (dt - last_cluster[-1][0]).days <= EVENT_COOLDOWN_DAYS:
                    last_cluster.append((dt, r))
                else:
                    selected.append(max(last_cluster, key=lambda x: x[1]["z"]))
                    last_cluster = [(dt, r)]

            if last_cluster:
                selected.append(max(last_cluster, key=lambda x: x[1]["z"]))

            for dt, r in selected:
                t = r["tone"]
                if pd.isna(t):
                    sentiment = "unknown"
                elif t <= -2:
                    sentiment = "negative"
                elif t >= 2:
                    sentiment = "positive"
                else:
                    sentiment = "neutral"

                rows.append({
                    "coin": coin,
                    "topic": topic,
                    "date": dt,
                    "article_count": float(r["count"]),
                    "event_z90": float(r["z"]),
                    "tone": float(t) if pd.notna(t) else np.nan,
                    "sentiment": sentiment,
                })

    return pd.DataFrame(rows).sort_values(["coin", "date", "topic"])


def _feature_state(price_features: pd.DataFrame, coin: str) -> pd.DataFrame:
    """
    Toestand vóór het event voor matched controls:
    30d momentum + 30d volatiliteit + 90d drawdown.
    """
    return pd.DataFrame({
        "mom30": price_features[f"{coin}_mom_30"],
        "vol30": price_features[f"{coin}_vol_30"],
        "dd90": price_features[f"{coin}_drawdown90"],
    })


def matched_controls(
    dt: pd.Timestamp,
    coin: str,
    events: pd.DataFrame,
    price_features: pd.DataFrame,
    k: int = 5,
) -> List[pd.Timestamp]:
    states = _feature_state(price_features, coin)
    if dt not in states.index or states.loc[dt].isna().any():
        return []

    # Controle uit dezelfde +/- 365 dagen om regimeverschillen te beperken.
    lo = dt - pd.Timedelta(days=365)
    hi = dt + pd.Timedelta(days=365)

    candidates = states.loc[(states.index >= lo) & (states.index <= hi)].dropna()

    # Eventdagen en +/-14 dagen eromheen uitsluiten.
    coin_events = pd.to_datetime(events.loc[events["coin"] == coin, "date"])
    banned = set()
    for e in coin_events:
        for d in range(-14, 15):
            banned.add((e + pd.Timedelta(days=d)).normalize())

    candidates = candidates.loc[~candidates.index.normalize().isin(banned)]
    if candidates.empty:
        return []

    # Standardiseer op kandidaatset.
    mu = candidates.mean()
    sd = candidates.std().replace(0, 1)
    target = (states.loc[dt] - mu) / sd
    zc = (candidates - mu) / sd
    dist = ((zc - target) ** 2).sum(axis=1) ** 0.5
    return dist.nsmallest(k).index.tolist()


def event_study(
    events: pd.DataFrame,
    price_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []

    for _, ev in events.iterrows():
        coin = ev["coin"]
        dt = pd.Timestamp(ev["date"])
        price = price_features[f"{coin}_close"]

        controls = matched_controls(dt, coin, events, price_features, k=5)

        for h in [1, 3, 7, 30]:
            if dt not in price.index:
                continue
            loc = price.index.get_loc(dt)
            if isinstance(loc, slice):
                continue
            if loc + h >= len(price):
                continue

            event_ret = price.iloc[loc + h] / price.iloc[loc] - 1

            c_rets = []
            for cdt in controls:
                if cdt not in price.index:
                    continue
                cloc = price.index.get_loc(cdt)
                if isinstance(cloc, slice) or cloc + h >= len(price):
                    continue
                c_rets.append(price.iloc[cloc + h] / price.iloc[cloc] - 1)

            control_mean = np.mean(c_rets) if c_rets else np.nan

            detail_rows.append({
                **ev.to_dict(),
                "horizon_days": h,
                "event_return": float(event_ret),
                "matched_control_return": float(control_mean) if pd.notna(control_mean) else np.nan,
                "excess_vs_matched": float(event_ret - control_mean) if pd.notna(control_mean) else np.nan,
                "n_controls": len(c_rets),
            })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, pd.DataFrame()

    summary_rows = []
    group_cols = ["coin", "topic", "sentiment", "horizon_days"]

    for keys, g in detail.groupby(group_cols):
        g = g.dropna(subset=["event_return"])
        excess = g["excess_vs_matched"].dropna()

        # Wilcoxon-achtige/non-parametrische test tegen gematchte controlreturns:
        # Mann-Whitney op eventreturns versus per-event gemiddelde controls.
        p = np.nan
        if len(g) >= 6 and g["matched_control_return"].notna().sum() >= 6:
            try:
                p = mannwhitneyu(
                    g["event_return"].dropna(),
                    g["matched_control_return"].dropna(),
                    alternative="two-sided"
                ).pvalue
            except Exception:
                p = np.nan

        summary_rows.append({
            "coin": keys[0],
            "topic": keys[1],
            "sentiment": keys[2],
            "horizon_days": keys[3],
            "n_events": len(g),
            "avg_event_return": g["event_return"].mean(),
            "median_event_return": g["event_return"].median(),
            "positive_return_rate": (g["event_return"] > 0).mean(),
            "avg_matched_control_return": g["matched_control_return"].mean(),
            "avg_excess_vs_matched": excess.mean() if len(excess) else np.nan,
            "median_excess_vs_matched": excess.median() if len(excess) else np.nan,
            "test_p_value": p,
        })

    summary = pd.DataFrame(summary_rows)

    valid = summary["test_p_value"].notna()
    summary["p_value_fdr"] = np.nan
    summary["significant_fdr_5pct"] = False
    if valid.any():
        rejected, corrected, _, _ = multipletests(
            summary.loc[valid, "test_p_value"].values,
            alpha=0.05,
            method="fdr_bh",
        )
        summary.loc[valid, "p_value_fdr"] = corrected
        summary.loc[valid, "significant_fdr_5pct"] = rejected

    return detail, summary


# ---------------------------------------------------------------------------
# Drie-modelvergelijking
# ---------------------------------------------------------------------------

def generic_news_features(df: pd.DataFrame) -> List[str]:
    cols = []
    for coin in core.COINS:
        for suffix in GENERIC_SUFFIXES:
            c = f"{coin}_news_{suffix}"
            if c in df.columns:
                cols.append(c)
    return cols


def event_news_features(df: pd.DataFrame) -> List[str]:
    """
    Alleen robuuste eventfeatures. We nemen bewust niet automatisch ieder door
    core aangemaakt topic-z-kenmerk mee: een categorie die jarenlang nul was
    kan daar een NaN-standaardafwijking hebben en zo onnodig historie weggooien.
    """
    cols = []
    suffixes = [
        "count",
        "share",
        "event_z30",
        "event_z90",
        "event_tone",
        "event_tone_z30",
        "event_intensity",
        "event_negative",
        "event_positive",
    ]
    for coin in core.COINS:
        for topic in EVENT_TOPICS:
            for suffix in suffixes:
                c = f"{coin}_news_{topic}_{suffix}"
                if c in df.columns:
                    cols.append(c)
    return cols


def compare_three_models(df: pd.DataFrame, output_dir: Path):
    base = core.base_feature_columns(df)
    generic = generic_news_features(df)
    event = event_news_features(df)

    rows = []
    latest = []

    for coin in core.COINS:
        for h in core.HORIZONS:
            y_class = f"{coin}_target_up_{h}"
            y_ret = f"{coin}_target_ret_{h}"

            # Exact dezelfde datums voor alle drie modellen.
            all_features = list(dict.fromkeys(base + generic + event))
            fair = df.dropna(subset=all_features + [y_class, y_ret])
            allowed = fair.index

            if len(allowed) < 700:
                raise RuntimeError(
                    f"Te weinig complete observaties voor {coin} {h}d: {len(allowed)}."
                )

            sets = {
                "price": base,
                "price_generic_news": list(dict.fromkeys(base + generic)),
                "price_event_news": all_features,
            }

            bts = {}
            mets = {}

            print(f"\n{coin} — {h} dagen")
            for label, feats in sets.items():
                bt = core.walk_forward(df, coin, h, feats, allowed_index=allowed)
                m = core.evaluate(bt)
                bts[label] = bt
                mets[label] = m
                bt.to_csv(output_dir / f"{coin}_{h}d_{label}_backtest.csv")
                print(
                    f"  {label:20s} AUC={m['auc']:.3f}  "
                    f"Brier={m['brier']:.4f}  Acc={m['accuracy']:.3f}"
                )

            # Vergelijk generiek nieuws tegen prijs en eventnieuws tegen generiek.
            comp_generic = core.paired_model_comparison(
                bts["price"], bts["price_generic_news"], coin, h
            )
            comp_event = core.paired_model_comparison(
                bts["price_generic_news"], bts["price_event_news"], coin, h
            )

            rows.append({
                "coin": coin,
                "horizon_days": h,
                "price_auc": mets["price"]["auc"],
                "generic_auc": mets["price_generic_news"]["auc"],
                "event_auc": mets["price_event_news"]["auc"],
                "delta_generic_vs_price_auc": mets["price_generic_news"]["auc"] - mets["price"]["auc"],
                "delta_event_vs_generic_auc": mets["price_event_news"]["auc"] - mets["price_generic_news"]["auc"],
                "price_brier": mets["price"]["brier"],
                "generic_brier": mets["price_generic_news"]["brier"],
                "event_brier": mets["price_event_news"]["brier"],
                "generic_vs_price_assessment": core.news_edge_label(comp_generic),
                "event_vs_generic_assessment": core.news_edge_label(comp_event),
                "event_vs_generic_brier_delta": comp_event.get("delta_brier_news_minus_base", np.nan),
                "event_vs_generic_brier_ci_low": comp_event.get("brier_delta_ci95_low", np.nan),
                "event_vs_generic_brier_ci_high": comp_event.get("brier_delta_ci95_high", np.nan),
            })

            for label, feats in sets.items():
                pred = core.latest_prediction(df, coin, h, feats, bts[label])
                latest.append({
                    "coin": coin,
                    "horizon_days": h,
                    "model": label,
                    **pred,
                })

    compare_df = pd.DataFrame(rows)
    latest_df = pd.DataFrame(latest)

    compare_df.to_csv(output_dir / "three_model_comparison.csv", index=False)
    latest_df.to_csv(output_dir / "latest_predictions_v3.csv", index=False)

    return compare_df, latest_df


def print_event_summary(summary: pd.DataFrame):
    print("\n" + "=" * 82)
    print("EVENT STUDY — STERKSTE HISTORISCHE RELATIES")
    print("=" * 82)

    if summary.empty:
        print("Geen eventresultaten.")
        return

    z = summary[summary["n_events"] >= 6].copy()
    if z.empty:
        print("Nog te weinig events per categorie voor een stabiele samenvatting.")
        return

    z["abs_excess"] = z["avg_excess_vs_matched"].abs()
    z = z.sort_values(
        ["significant_fdr_5pct", "abs_excess", "n_events"],
        ascending=[False, False, False],
    )

    for _, r in z.head(20).iterrows():
        sig = "*" if bool(r["significant_fdr_5pct"]) else ""
        print(
            f"{r['coin']:3s} {r['topic']:24s} {r['sentiment']:8s} "
            f"{int(r['horizon_days']):2d}d  n={int(r['n_events']):3d}  "
            f"event={pct(r['avg_event_return'])}  "
            f"excess={pct(r['avg_excess_vs_matched'])}  "
            f"FDR-p={r['p_value_fdr']:.4f}{sig}"
            if pd.notna(r["p_value_fdr"])
            else
            f"{r['coin']:3s} {r['topic']:24s} {r['sentiment']:8s} "
            f"{int(r['horizon_days']):2d}d  n={int(r['n_events']):3d}  "
            f"event={pct(r['avg_event_return'])}  "
            f"excess={pct(r['avg_excess_vs_matched'])}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-news-refresh", action="store_true")
    parser.add_argument("--cache-dir", default="cache_v3")
    parser.add_argument("--output-dir", default="output_v3")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("BTC/ETH Predictor v3 — prijs + nieuws + eventcategorieën")
    print("=" * 82)

    print("1/7 Koersdata...")
    price = core.download_price_data()
    price_features = core.add_price_features(price)

    news_start = max(price.index.min(), core.GDELT_START)
    news_end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)

    print("2/7 Algemeen nieuws + categorievolume...")
    raw_news = core.load_or_update_news(
        cache_dir=cache_dir,
        start=news_start,
        end=news_end,
        force=args.force_news_refresh,
    )
    raw_news.to_csv(output_dir / "raw_news_with_event_volume.csv", index_label="date")

    print("3/7 Categorie-specifieke tone...")
    raw_event_tone = load_event_tone_cache(
        cache_dir=cache_dir,
        start=news_start,
        end=news_end,
        force=args.force_news_refresh,
    )
    raw_event_tone.to_csv(output_dir / "raw_event_tone.csv", index_label="date")

    print("4/7 Features construeren...")
    model_df = core.add_news_features(price_features, raw_news)
    model_df = add_event_features(model_df, raw_news, raw_event_tone)

    print("5/7 Eventdetectie + event study...")
    events = detect_events(price_features, raw_news, raw_event_tone)
    events.to_csv(output_dir / "detected_news_events.csv", index=False)

    event_detail, event_summary = event_study(events, price_features)
    event_detail.to_csv(output_dir / "event_study_detail.csv", index=False)
    event_summary.to_csv(output_dir / "event_study_summary.csv", index=False)
    print_event_summary(event_summary)

    print("\n6/7 Richting/correlatie-onderzoek...")
    # v2-correlaties werken op de generieke + categorie-share features.
    corr = core.correlation_study(model_df)
    corr.to_csv(output_dir / "news_correlations_v3.csv", index=False)

    granger = core.granger_study(model_df)
    granger.to_csv(output_dir / "granger_tests_v3.csv", index=False)
    core.summarize_relationships(corr, granger)

    print("\n7/7 Drie modellen walk-forward vergelijken...")
    compare_df, latest_df = compare_three_models(model_df, output_dir)

    print("\n" + "=" * 82)
    print("SAMENVATTING MODELVERGELIJKING")
    print("=" * 82)
    print(compare_df.to_string(index=False))

    print("\nActuele modeluitkomsten:")
    for _, r in latest_df.iterrows():
        print(
            f"{r['coin']} {int(r['horizon_days'])}d {r['model']:20s} "
            f"P(up)={pct(r['prob_up'])}  "
            f"E[ret]={pct(r['expected_return'])}  "
            f"80%={pct(r['range80_low'])}..{pct(r['range80_high'])}"
        )

    print("\nKlaar.")
    print(f"Alle CSV-uitkomsten: {output_dir.resolve()}")
    print(
        "\nBelangrijk: een eventcategorie krijgt pas praktische waarde als "
        "de event study én het out-of-sample EVENT-model consistent beter zijn. "
        "Een losse significante correlatie is onvoldoende."
    )


if __name__ == "__main__":
    main()
