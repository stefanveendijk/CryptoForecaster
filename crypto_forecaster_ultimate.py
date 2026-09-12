from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, Config
import providers
import features as feat
import modeling

# Reused news/event engine
import news_core
import event_engine

def p(x):
    if x is None or pd.isna(x): return "n.v.t."
    return f"{100*x:.1f}%"

def source_quality(sources):
    rows=[]
    for name,d in sources.items():
        if d is None or d.empty:
            rows.append({"source":name,"status":"missing","start":None,"end":None,"rows":0,"columns":0})
        else:
            rows.append({
                "source":name,"status":"ok","start":str(d.index.min().date()),
                "end":str(d.index.max().date()),"rows":len(d),"columns":len(d.columns),
                "missing_pct":float(d.isna().mean().mean())
            })
    return pd.DataFrame(rows)

def add_news(df, cfg, cache_dir, output_dir, force=False):
    if not cfg.enable_news:
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        # Align the v3 module's topic taxonomy.
        news_core.TOPIC_QUERIES = event_engine.EVENT_TOPICS
        event_engine.core = news_core

        start=max(df.index.min(), news_core.GDELT_START)
        end=pd.Timestamp.today().normalize()-pd.Timedelta(days=1)

        raw_news=news_core.load_or_update_news(cache_dir/"gdelt",start,end,force)
        raw_tone=event_engine.load_event_tone_cache(cache_dir/"gdelt",start,end,force)

        # news_core.add_news_features expects price-target columns from its own naming scheme.
        # We only need the generated news columns; feed a minimal carrier frame and strip it back.
        carrier=pd.DataFrame(index=df.index)
        # Create placeholder BTC/ETH columns used by news feature constructor only for joins.
        for coin in ["BTC","ETH"]:
            carrier[f"{coin}_close"]=df[f"{coin}_close"]
            carrier[f"{coin}_ret1"]=np.log(df[f"{coin}_close"]).diff()
        newsframe=news_core.add_news_features(carrier,raw_news)
        newscols=[c for c in newsframe if "_news_" in c]
        df=df.join(newsframe[newscols],how="left")

        # Add richer event tone/z/intensity via event engine.
        enriched=event_engine.add_event_features(df,raw_news,raw_tone)
        df=enriched

        events=event_engine.detect_events(
            # event_engine expects v2-style price feature names. Make adapter.
            pd.DataFrame({
                "BTC_close":df["BTC_close"],"ETH_close":df["ETH_close"],
                "BTC_mom_30":df["TECH_BTC_mom30"],"ETH_mom_30":df["TECH_ETH_mom30"],
                "BTC_vol_30":df["TECH_BTC_vol30"],"ETH_vol_30":df["TECH_ETH_vol30"],
                "BTC_drawdown90":df["TECH_BTC_dd90"],"ETH_drawdown90":df["TECH_ETH_dd90"],
            },index=df.index),
            raw_news,raw_tone
        )
        # v3 event study has strict column naming adapter needs; skip here because v4 model
        # uses event features directly. Persist raw detected events.
        events.to_csv(output_dir/"detected_events.csv",index=False)
        return df,raw_news,raw_tone,events
    except Exception as e:
        print("WAARSCHUWING nieuwslaag:",e)
        traceback.print_exc(limit=1)
        return df,pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

def run(cfg:Config, workdir:Path, force=False):
    cache=workdir/"cache"
    outdir=workdir/"output"
    cache.mkdir(parents=True,exist_ok=True)
    outdir.mkdir(parents=True,exist_ok=True)

    print("="*90)
    print("CRYPTO FORECASTER ULTIMATE — leakage-aware multi-expert ensemble")
    print("="*90)

    print("\n1/6 Data verzamelen...")
    sources=providers.gather_all_nonnews(cfg,cache,force=force)
    q=source_quality(sources)
    q.to_csv(outdir/"data_quality.csv",index=False)
    print(q.to_string(index=False))

    print("\n2/6 Features bouwen...")
    df=feat.build_features(sources,cfg)

    print("\n3/6 Nieuws + gebeurtenissen...")
    df,raw_news,raw_tone,events=add_news(df,cfg,cache,outdir,force=force)

    # Persist dataset metadata, not full raw data by default.
    groups=feat.feature_groups(df)
    manifest=[]
    for g,cols in groups.items():
        manifest.append({"group":g,"n_features":len(cols),
                         "features":";".join(cols)})
    pd.DataFrame(manifest).to_csv(outdir/"feature_manifest.csv",index=False)

    experts=feat.build_experts(df)
    with open(outdir/"experts.json","w",encoding="utf-8") as f:
        json.dump(experts,f,ensure_ascii=False,indent=2)

    print("\nFeaturegroepen:")
    for k,v in groups.items():
        print(f"  {k:14s}: {len(v):4d}")
    print("Experts:")
    for k,v in experts.items():
        print(f"  {k:20s}: {len(v):4d} kandidaatfeatures")

    print("\n4/6 Walk-forward experts...")
    all_summary=[]
    latest_rows=[]

    for coin in cfg.coins:
        for h in cfg.horizons:
            print(f"\n--- {coin} horizon {h}d ---")
            bts={}
            used={}
            for name,cols in experts.items():
                try:
                    bt,used_features=modeling.walk_forward_expert(df,coin,h,cols,cfg,name)
                    if bt.empty:
                        print(f"  {name:20s}: overgeslagen (te weinig bruikbare historie)")
                        continue
                    bts[name]=bt
                    used[name]=used_features
                    m=modeling.metrics(bt)
                    print(f"  {name:20s}: n={m['n']:4d} AUC={m['auc']:.3f} "
                          f"Brier={m['brier']:.4f} MAE={m['mae_return']:.4f}")
                    row={"coin":coin,"horizon":h,"expert":name,**m}
                    # Lockbox = latest OOS period, never a training shortcut.
                    lock=bt.tail(cfg.lockbox_days)
                    lm=modeling.metrics(lock)
                    row.update({f"lockbox_{k}":v for k,v in lm.items()})
                    all_summary.append(row)
                    bt.to_csv(outdir/f"{coin}_{h}d_{name}_oos.csv")
                except Exception as e:
                    print(f"  {name:20s}: fout: {e}")

            if not bts:
                continue

            meta=modeling.meta_ensemble(bts,cfg)
            if not meta.empty:
                mm=modeling.metrics(meta)
                sm=modeling.strategy_metrics(meta,df[f"{coin}_close"],cfg)
                print(f"  {'META-ENSEMBLE':20s}: n={mm['n']:4d} AUC={mm['auc']:.3f} "
                      f"Brier={mm['brier']:.4f} MAE={mm['mae_return']:.4f}")
                if sm:
                    print(f"  {'Strategy check':20s}: CAGR={p(sm['strategy_cagr'])} "
                          f"BH={p(sm['buyhold_cagr'])} Sharpe={sm['strategy_sharpe']:.2f} "
                          f"MaxDD={p(sm['max_drawdown'])}")
                meta.to_csv(outdir/f"{coin}_{h}d_META_oos.csv")
                all_summary.append({"coin":coin,"horizon":h,"expert":"META",**mm,**{
                    f"strategy_{k}":v for k,v in sm.items()
                }})

            # Latest predictions from every viable expert.
            current=[]
            for name,cols in experts.items():
                try:
                    lp=modeling.latest_expert_prediction(df,coin,h,cols,cfg,name)
                    if lp:
                        current.append(lp)
                except Exception:
                    pass
            lm=modeling.latest_meta(current,bts,cfg)
            if lm:
                current_price=float(df[f"{coin}_close"].loc[lm["date"]])
                lm["current_price"]=current_price
                lm["expected_price"]=current_price*(1+lm["pred_ret"])
                lm["low80_price"]=current_price*(1+lm["low80"])
                lm["high80_price"]=current_price*(1+lm["high80"])
                latest_rows.append({"coin":coin,"horizon":h,"model":"META",**lm})
                print(f"  LAATSTE META: P(up)={p(lm['prob_up'])} "
                      f"E[ret]={p(lm['pred_ret'])} 80%={p(lm['low80'])}..{p(lm['high80'])} "
                      f"confidence={lm.get('confidence_score',float('nan')):.2f}")
                print("  Gewichten:",", ".join(f"{k}={v:.2f}" for k,v in lm["weights"].items()))
            for lp in current:
                latest_rows.append({"coin":coin,"horizon":h,"model":lp["expert"],**lp})

    print("\n5/6 Resultaten opslaan...")
    summary=pd.DataFrame(all_summary)
    latest=pd.DataFrame(latest_rows)
    summary.to_csv(outdir/"model_summary.csv",index=False)
    if not latest.empty:
        latest["weights"]=latest.get("weights",pd.Series(dtype=object)).apply(
            lambda z: json.dumps(z) if isinstance(z,dict) else z
        )
    latest.to_csv(outdir/"latest_forecasts.csv",index=False)

    # Current snapshot of key raw features.
    last=df.index.max()
    snapshot=df.loc[last].dropna()
    snapshot=snapshot[~snapshot.index.str.startswith("TARGET_")]
    snapshot.rename("value").to_csv(outdir/"latest_feature_snapshot.csv")

    print("\n6/6 HTML-rapport...")
    make_report(outdir,summary,latest,q,groups,events)

    print("\nKLAAR")
    print("Output:",outdir.resolve())
    return outdir

def make_report(outdir,summary,latest,quality,groups,events):
    def table(d,n=30):
        if d is None or d.empty:return "<p>Geen data.</p>"
        return d.head(n).to_html(index=False,border=0,classes="data")
    latest_meta=latest[latest["model"]=="META"].copy() if not latest.empty else latest
    if latest_meta is not None and not latest_meta.empty:
        for c in ["prob_up","pred_ret","low80","high80"]:
            latest_meta[c]=latest_meta[c].map(lambda x:f"{100*x:.1f}%" if pd.notna(x) else "")
    best=summary[summary.expert=="META"].copy() if not summary.empty else summary
    html=f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Crypto Forecaster Ultimate</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;max-width:1200px;color:#222}}
h1,h2{{margin-top:28px}} table.data{{border-collapse:collapse;width:100%;font-size:13px}}
.data th,.data td{{border:1px solid #ddd;padding:6px;text-align:right}}
.data th:first-child,.data td:first-child{{text-align:left}} .note{{background:#f5f5f5;padding:14px}}
</style></head><body>
<h1>Crypto Forecaster Ultimate</h1>
<div class="note"><b>Onderzoeksmodel, geen rendementsbelofte.</b> Alle modelkeuzes zijn
gericht op out-of-sample voorspelkracht, leakage-preventie en robuuste onzekerheidsrapportage.</div>
<h2>Actuele meta-voorspellingen</h2>{table(latest_meta,20)}
<h2>Out-of-sample meta-prestaties</h2>{table(best,30)}
<h2>Databronnen</h2>{table(quality,30)}
<h2>Featuregroepen</h2><ul>{''.join(f'<li>{k}: {len(v)} features</li>' for k,v in groups.items())}</ul>
<h2>Gedetecteerde nieuwsevents</h2>{table(events.tail(30) if events is not None and not events.empty else pd.DataFrame(),30)}
<h2>Methodologische waarborgen</h2>
<ul>
<li>Expanding walk-forward backtesting.</li>
<li>Purging/embargo: target-uitkomst moet bekend zijn vóór gebruik als training.</li>
<li>Exogene bronnen worden vertraagd volgens praktische beschikbaarheid.</li>
<li>Experts met korte historie verkorten niet de historie van het hele model.</li>
<li>Modelgewichten komen uit eerdere out-of-sample Brier-score, niet uit toekomstige data.</li>
<li>80%-banden komen uit historische validatiefouten.</li>
<li>Laatste 365 OOS-dagen worden apart als lockbox gerapporteerd.</li>
</ul>
</body></html>"""
    (outdir/"report.html").write_text(html,encoding="utf-8")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--force-refresh",action="store_true")
    ap.add_argument("--deep",action="store_true")
    ap.add_argument("--no-news",action="store_true")
    ap.add_argument("--google-trends",action="store_true")
    ap.add_argument("--workdir",default="ultimate_run")
    args=ap.parse_args()
    cfg=DEFAULT_CONFIG
    if args.deep:
        cfg.mode="deep"
        cfg.retrain_every_days=30
    if args.no_news: cfg.enable_news=False
    if args.google_trends: cfg.enable_google_trends=True
    run(cfg,Path(args.workdir),force=args.force_refresh)
