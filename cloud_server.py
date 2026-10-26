from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import trading_strategy
import strategy_audit
import portfolio_strategy
import forecast_proof
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
WORKDIR = DATA_DIR / "ultimate_run"
OUTPUT_DIR = WORKDIR / "output"
STATE_FILE = DATA_DIR / "cloud_state.json"

MODEL_HOUR_UTC = int(os.getenv("MODEL_HOUR_UTC", "4"))
MODEL_MINUTE_UTC = int(os.getenv("MODEL_MINUTE_UTC", "15"))
MODEL_DEEP = os.getenv("MODEL_DEEP", "false").lower() in {"1","true","yes"}
MODEL_NEWS = os.getenv("MODEL_NEWS", "true").lower() not in {"0","false","no"}
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
AUTO_RUN = os.getenv("AUTO_RUN_MODEL", "true").lower() not in {"0","false","no"}
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() in {"1","true","yes"}

app = FastAPI(title="Crypto Forecaster API", version="4.2-local")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

model_lock = threading.Lock()
run_thread: threading.Thread | None = None
kraken_cache: dict[str, Any] = {"ts": 0.0, "data": None}
KRAKEN_CACHE_SECONDS = 30.0

# The free GitHub Pages publication is the shared forecast source for the laptop.
# Strategy and audit endpoints remain local and MUST be labelled separately.
SHARED_FORECAST_URL = os.getenv(
    "SHARED_FORECAST_URL",
    "https://stefanveendijk.github.io/CryptoForecaster/data/dashboard.json",
)
SHARED_FORECAST_CACHE_SECONDS = 300.0
shared_forecast_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
shared_forecast_lock = threading.Lock()
SHARED_HISTORY_BASE = "https://stefanveendijk.github.io/CryptoForecaster/data/history"
shared_history_cache: dict[str, dict[str, Any]] = {}
shared_history_lock = threading.Lock()

def utcnow():
    return datetime.now(timezone.utc)

def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {
            "running": False,
            "last_started": None,
            "last_finished": None,
            "last_success": None,
            "last_error": None,
            "last_exit_code": None,
        }

def save_state(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def clean_number(v: Any):
    if v is None:
        return None
    try:
        n=float(v)
        if np.isnan(n) or np.isinf(n):
            return None
        return n
    except Exception:
        return None

def parse_weights(v: Any):
    if isinstance(v, dict):
        return v
    if not isinstance(v, str) or not v.strip():
        return {}
    try:
        x=json.loads(v)
        return {str(k):float(val) for k,val in x.items()}
    except Exception:
        return {}

def demo_payload():
    now=utcnow().isoformat()
    return {
        "updatedAt": now,
        "source": "initializing",
        "forecasts": [],
        "metrics": [],
        "events": [],
        "message": "Het model heeft nog geen eerste volledige run afgerond."
    }

def run_model(reason: str = "scheduled"):
    global run_thread
    if not model_lock.acquire(blocking=False):
        return

    state=load_state()
    state.update({
        "running": True,
        "last_started": utcnow().isoformat(),
        "last_error": None,
        "reason": reason,
    })
    save_state(state)

    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        cmd=[
            sys.executable,
            "-u",
            str(APP_DIR/"crypto_forecaster_ultimate.py"),
            "--workdir", str(WORKDIR),
        ]
        if MODEL_DEEP:
            cmd.append("--deep")
        if not MODEL_NEWS:
            cmd.append("--no-news")

        env=os.environ.copy()
        env.setdefault("OMP_NUM_THREADS","2")
        env.setdefault("OPENBLAS_NUM_THREADS","2")
        env.setdefault("MKL_NUM_THREADS","2")

        log_path=DATA_DIR/"last_model_run.log"
        with log_path.open("w",encoding="utf-8",errors="replace") as log:
            log.write(f"START {utcnow().isoformat()} reason={reason}\n")
            log.write("CMD: "+" ".join(cmd)+"\n\n")
            log.flush()
            proc=subprocess.run(
                cmd,
                cwd=str(APP_DIR),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        state=load_state()
        state["last_exit_code"]=proc.returncode
        state["last_finished"]=utcnow().isoformat()
        state["running"]=False
        if proc.returncode==0 and (OUTPUT_DIR/"latest_forecasts.csv").exists():
            state["last_success"]=state["last_finished"]
            state["last_error"]=None
        else:
            state["last_error"]=f"Model exit code {proc.returncode}. Bekijk /api/v1/log-tail."
        save_state(state)
    except Exception as e:
        state=load_state()
        state.update({
            "running": False,
            "last_finished": utcnow().isoformat(),
            "last_error": repr(e),
        })
        save_state(state)
    finally:
        model_lock.release()

def start_model_thread(reason: str):
    global run_thread
    if run_thread is not None and run_thread.is_alive():
        return False
    run_thread=threading.Thread(target=run_model,args=(reason,),daemon=True)
    run_thread.start()
    return True

def scheduler_loop():
    time.sleep(15)
    while True:
        try:
            state=load_state()
            now=utcnow()
            latest=OUTPUT_DIR/"latest_forecasts.csv"

            if AUTO_RUN and not latest.exists() and not state.get("running"):
                start_model_thread("initial")

            if AUTO_RUN and latest.exists() and not state.get("running"):
                last=state.get("last_success")
                last_dt=None
                try:
                    last_dt=datetime.fromisoformat(last) if last else None
                except Exception:
                    pass
                due=(now.hour>MODEL_HOUR_UTC or
                     (now.hour==MODEL_HOUR_UTC and now.minute>=MODEL_MINUTE_UTC))
                already_today=(last_dt is not None and last_dt.date()==now.date())
                if due and not already_today:
                    start_model_thread("daily")
        except Exception:
            pass
        time.sleep(300)

@app.on_event("startup")
def startup():
    DATA_DIR.mkdir(parents=True,exist_ok=True)

    # A process restart/redeploy kills any previous model subprocess.
    # If the persistent state still says "running", it is stale and would
    # otherwise prevent the new deployment from ever starting a fresh run.
    state = load_state()
    if state.get("running"):
        state["running"] = False
        state["last_error"] = "Vorige modelrun is onderbroken door restart/redeploy; automatische herstart volgt."
        save_state(state)

    threading.Thread(target=scheduler_loop,daemon=True).start()

@app.get("/", include_in_schema=False)
def root():
    page=APP_DIR/"local_dashboard.html"
    if page.exists():
        return FileResponse(page,media_type="text/html",headers={"Cache-Control":"no-store, no-cache, must-revalidate, max-age=0","Pragma":"no-cache","Expires":"0"})
    return {
        "name":"Crypto Forecaster",
        "ok":True,
        "health":"/api/v1/health",
        "dashboard":"/api/v1/dashboard",
    }

@app.get("/api/v1/health")
def health():
    state=load_state()
    return {
        "ok": True,
        "cloud": not LOCAL_MODE,
        "local": LOCAL_MODE,
        "modelOutputExists": (OUTPUT_DIR/"latest_forecasts.csv").exists(),
        "modelRunning": bool(state.get("running")),
        "lastSuccess": state.get("last_success"),
        "lastError": state.get("last_error"),
        "dataDir": str(DATA_DIR),
    }

@app.get("/api/v1/run-status")
def run_status():
    return load_state()

@app.get("/api/v1/log-tail")
def log_tail(lines:int=Query(default=80,ge=10,le=300)):
    p=DATA_DIR/"last_model_run.log"
    if not p.exists():
        return {"lines":[]}
    text=p.read_text(encoding="utf-8",errors="replace").splitlines()
    return {"lines":text[-lines:]}

@app.post("/api/v1/run-now")
def run_now(x_admin_key: str | None = Header(default=None)):
    if not ADMIN_KEY:
        raise HTTPException(403,"ADMIN_KEY is niet ingesteld.")
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403,"Onjuiste admin key.")
    started=start_model_thread("manual")
    return {"ok":True,"started":started,"state":load_state()}

def _kraken_pair_quote(symbol: str) -> dict:
    r = requests.get(
        "https://api.kraken.com/0/public/PreTrade",
        params={"symbol": symbol},
        headers={"Accept": "application/json", "User-Agent": "CryptoForecaster/4.2-local"},
        timeout=8,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError("; ".join(str(x) for x in payload["error"]))

    result = payload.get("result") or {}
    bids = result.get("bids") or []
    asks = result.get("asks") or []
    if not bids or not asks:
        raise RuntimeError(f"Geen bied/laatprijs ontvangen voor {symbol}")

    bid = clean_number(bids[0].get("price"))
    ask = clean_number(asks[0].get("price"))
    if bid is None or ask is None:
        raise RuntimeError(f"Ongeldige Kraken-prijs voor {symbol}")

    mid = (bid + ask) / 2.0
    spread = ask - bid
    spread_pct = spread / mid if mid else None
    return {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "spreadPct": spread_pct,
    }


@app.get("/api/v1/kraken")
def kraken_prices():
    now = time.time()
    cached = kraken_cache.get("data")
    cached_ts = float(kraken_cache.get("ts") or 0.0)
    if cached is not None and now - cached_ts < KRAKEN_CACHE_SECONDS:
        return cached

    try:
        data = {
            "source": "Kraken",
            "currency": "EUR",
            "updatedAt": utcnow().isoformat(),
            "BTC": _kraken_pair_quote("BTC/EUR"),
            "ETH": _kraken_pair_quote("ETH/EUR"),
        }
        kraken_cache["ts"] = now
        kraken_cache["data"] = data
        return data
    except Exception as e:
        if cached is not None:
            stale = dict(cached)
            stale["stale"] = True
            stale["warning"] = f"Actuele Kraken-prijs tijdelijk niet bereikbaar: {e}"
            return stale
        raise HTTPException(503, f"Kraken-prijzen tijdelijk niet beschikbaar: {e}")


def load_shared_forecast() -> dict[str, Any] | None:
    """Use one published forecast source, with a short bounded HTTP timeout."""
    if not LOCAL_MODE:
        return None
    now = time.monotonic()
    with shared_forecast_lock:
        cached = shared_forecast_cache["payload"]
        if cached is not None and now - shared_forecast_cache["ts"] < SHARED_FORECAST_CACHE_SECONDS:
            return dict(cached)
    try:
        response = requests.get(
            SHARED_FORECAST_URL,
            headers={"Accept": "application/json", "User-Agent": "CryptoForecaster/4.2-local"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("source") != "github-pages":
            return None
        forecasts = payload.get("forecasts")
        if not isinstance(forecasts, list) or not forecasts or not payload.get("updatedAt"):
            return None
        if any(not isinstance(f, dict) or "probUp" not in f or "coin" not in f for f in forecasts):
            return None
    except (requests.RequestException, ValueError, TypeError):
        return None
    with shared_forecast_lock:
        shared_forecast_cache.update(ts=now, payload=payload)
    return dict(payload)

@app.get("/api/v1/dashboard")
def dashboard():
    shared = load_shared_forecast()
    if shared is not None:
        shared["forecastOrigin"] = "github-pages"
        shared["strategyOrigin"] = "local-model"
        shared["warning"] = (
            "Voorspellingen komen uit het gedeelde online model. Handelsstrategie, "
            "strategie-audit en portefeuilleverdeling gebruiken nog lokale modelresultaten "
            "en kunnen daardoor afwijken."
        )
        return shared
    latest_path=OUTPUT_DIR/"latest_forecasts.csv"
    if not latest_path.exists():
        result = demo_payload()
        if LOCAL_MODE:
            result["warning"] = "Online model niet bereikbaar; er zijn ook geen lokale voorspellingen beschikbaar."
        return result

    latest=pd.read_csv(latest_path)
    latest=latest[latest["model"].astype(str).str.upper()=="META"].copy()
    forecasts=[]
    for _,r in latest.iterrows():
        forecasts.append({
            "coin":str(r.get("coin","")).upper(),
            "horizon":int(r.get("horizon",0)),
            "date":str(r.get("date","")),
            "probUp":clean_number(r.get("prob_up")) or 0,
            "predReturn":clean_number(r.get("pred_ret")) or 0,
            "low80":clean_number(r.get("low80")) or 0,
            "high80":clean_number(r.get("high80")) or 0,
            "currentPrice":clean_number(r.get("current_price")),
            "expectedPrice":clean_number(r.get("expected_price")),
            "low80Price":clean_number(r.get("low80_price")),
            "high80Price":clean_number(r.get("high80_price")),
            "confidenceScore":clean_number(r.get("confidence_score")),
            "oodFraction":clean_number(r.get("ood_fraction")),
            "expertDisagreement":clean_number(r.get("expert_disagreement")),
            "weights":parse_weights(r.get("weights")),
        })

    metrics=[]
    mp=OUTPUT_DIR/"model_summary.csv"
    if mp.exists():
        m=pd.read_csv(mp)
        m=m[m["expert"].astype(str).str.upper()=="META"]
        for _,r in m.iterrows():
            metrics.append({
                "coin":str(r.get("coin","")).upper(),
                "horizon":int(r.get("horizon",0)),
                "expert":"META",
                "auc":clean_number(r.get("auc")),
                "brier":clean_number(r.get("brier")),
                "accuracy":clean_number(r.get("accuracy")),
                "balancedAccuracy":clean_number(r.get("balanced_accuracy")),
                "maeReturn":clean_number(r.get("mae_return")),
            })

    events=[]
    ep=OUTPUT_DIR/"detected_events.csv"
    if ep.exists():
        e=pd.read_csv(ep).tail(50)
        for _,r in e.iloc[::-1].iterrows():
            events.append({
                "coin":str(r.get("coin","")).upper(),
                "topic":str(r.get("topic","")),
                "date":str(r.get("date","")),
                "tone":clean_number(r.get("tone")),
                "sentiment":str(r.get("sentiment","")),
                "articleCount":clean_number(r.get("article_count")),
                "eventZ90":clean_number(r.get("event_z90")),
            })

    updated=max((f["date"] for f in forecasts),default=utcnow().isoformat())
    return {
        "updatedAt":updated,
        "source":"local-fallback" if LOCAL_MODE else "live",
        "forecastOrigin":"local-fallback" if LOCAL_MODE else "local-model",
        "strategyOrigin":"local-model",
        "baseCurrency":"USD",
        "warning":("Online model niet bereikbaar: tijdelijk lokale voorspellingen getoond. "
                   "Deze cijfers kunnen afwijken van GitHub Pages.") if LOCAL_MODE else None,
        "forecasts":forecasts,
        "metrics":metrics,
        "events":events,
    }

def load_price_history(allow_download: bool = True) -> pd.DataFrame:
    p=OUTPUT_DIR/"price_history.csv"
    try:
        if p.exists():
            d=pd.read_csv(p,parse_dates=["date"]).set_index("date").sort_index()
            d.index=pd.to_datetime(d.index).tz_localize(None).normalize()
            return d
    except Exception:
        pass

    if not allow_download:
        return pd.DataFrame()

    # Backfill voor een bestaande lokale installatie die al een volledige modelrun
    # heeft gedaan voordat price_history.csv werd toegevoegd. Dit gebeurt maximaal
    # één keer; daarna wordt het lokaal opgeslagen.
    try:
        from config import DEFAULT_CONFIG
        import providers
        d=providers.fetch_yfinance_crypto(DEFAULT_CONFIG)
        keep=[x for x in ["BTC_close","ETH_close"] if x in d.columns]
        if keep:
            d=d[keep].copy()
            d.index.name="date"
            OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
            d.to_csv(p)
            return d
    except Exception:
        pass
    return pd.DataFrame()


@app.get("/api/v1/strategy")
def strategy(
    coin:str=Query(pattern="^(BTC|ETH)$"),
    current_position:float=Query(default=0.0),
):
    if current_position not in {0.0,0.5,1.0}:
        raise HTTPException(400,"current_position moet 0, 0.5 of 1 zijn.")
    prices=load_price_history(allow_download=False)
    shared = load_shared_forecast()
    latest_override = None
    if shared:
        selected = {}
        for f in shared.get("forecasts", []):
            if str(f.get("coin", "")).upper() != coin.upper():
                continue
            try:
                horizon = int(f["horizon"])
            except (KeyError, ValueError, TypeError):
                continue
            if horizon not in trading_strategy.HORIZONS:
                continue
            selected[horizon] = {
                "date": str(f.get("date", "")),
                "prob_up": clean_number(f.get("probUp")),
                "pred_ret": clean_number(f.get("predReturn")),
                "confidence": clean_number(f.get("confidenceScore")),
                "expected_price": clean_number(f.get("expectedPrice")),
                "low80_price": clean_number(f.get("low80Price")),
                "high80_price": clean_number(f.get("high80Price")),
            }
        # Even incomplete published data must not silently fall back to an old
        # local BUY/SELL signal. The signal builder will safely block it.
        latest_override = selected
    try:
        return trading_strategy.build_strategy(
            OUTPUT_DIR, prices, coin, current_position=current_position,
            latest_override=latest_override,
            signal_source="github-pages" if shared else "local-model",
        )
    except Exception as e:
        raise HTTPException(500,f"Strategieberekening mislukt: {e}")

@app.get("/api/v1/portfolio-strategy")
def portfolio_strategy_endpoint():
    try:
        return portfolio_strategy.build_portfolio_strategy(OUTPUT_DIR)
    except Exception as e:
        raise HTTPException(500,f"Portefeuillestrategie mislukt: {e}")

@app.get("/api/v1/strategy-audit")
def strategy_audit_endpoint(
    coin:str=Query(pattern="^(BTC|ETH)$"),
):
    prices=load_price_history(allow_download=False)
    try:
        return strategy_audit.build_audit(OUTPUT_DIR,prices,coin)
    except Exception as e:
        raise HTTPException(500,f"Strategie-audit mislukt: {e}")

@app.get("/api/v1/proof")
def proof():
    try:
        prices=load_price_history(allow_download=False)
        return forecast_proof.build_summary(OUTPUT_DIR,prices)
    except Exception as e:
        raise HTTPException(500,f"Live bewijslaag mislukt: {e}")


def load_shared_history(coin: str, horizon: int) -> dict[str, Any] | None:
    """Retrieve OOS model history from the same publication as shared forecasts."""
    if not LOCAL_MODE or coin not in {"BTC", "ETH"} or horizon not in {1, 7, 30, 90}:
        return None
    key = f"{coin}_{horizon}"
    now = time.monotonic()
    with shared_history_lock:
        cached = shared_history_cache.get(key)
        if cached and now - cached["ts"] < SHARED_FORECAST_CACHE_SECONDS:
            return dict(cached["data"])
    try:
        response = requests.get(
            f"{SHARED_HISTORY_BASE}/{key}.json",
            headers={"Accept": "application/json", "User-Agent": "CryptoForecaster/4.2-local"},
            timeout=10,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict) or raw.get("coin") != coin or raw.get("horizon") != horizon:
            return None
        rows = raw.get("points")
        if not isinstance(rows, list) or not rows:
            return None
        points = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("date"):
                continue
            try:
                dt = pd.Timestamp(row["date"]).tz_localize(None).normalize()
            except (ValueError, TypeError):
                continue
            prediction = clean_number(row.get("predReturn"))
            outcome = clean_number(row.get("actualReturn"))
            if prediction is None or outcome is None:
                continue
            points.append({
                "date": str(dt.date()),
                "targetDate": str((dt + pd.Timedelta(days=horizon)).date()),
                "probUp": clean_number(row.get("probUp")),
                "predReturn": prediction,
                "actualReturn": outcome,
                "startPrice": None,
                "predictedPrice": None,
                "actualPrice": None,
                "low80Price": None,
                "high80Price": None,
            })
        if not points:
            return None
        points.sort(key=lambda row: row["date"])
        result = {
            "coin": coin, "horizon": horizon,
            "source": "github-pages-oos",
            "points": points[-500:],
            "lastHistoryDate": points[-1]["date"],
            "lastObservedTargetDate": points[-1]["targetDate"],
            "note": "Afgeronde historische out-of-sample toetsresultaten; geen toekomstige voorspellingen.",
        }
    except (requests.RequestException, ValueError, TypeError):
        return None
    with shared_history_lock:
        shared_history_cache[key] = {"ts": now, "data": result}
    return dict(result)


@app.get("/api/v1/history")
def history(
    coin:str=Query(pattern="^(BTC|ETH)$"),
    horizon:int=Query(ge=1,le=365),
):
    shared = load_shared_history(coin, horizon)
    if shared is not None:
        return shared
    p=OUTPUT_DIR/f"{coin}_{horizon}d_META_oos.csv"
    if not p.exists():
        raise HTTPException(404,"Nog geen history-bestand voor deze combinatie.")
    d=pd.read_csv(p)
    if "date" not in d.columns:
        d=d.rename(columns={d.columns[0]:"date"})
    d=d.tail(500)
    prices=load_price_history()
    price_col=f"{coin}_close"
    points=[]
    for _,r in d.iterrows():
        date_raw=r.get("date")
        try:
            dt=pd.Timestamp(date_raw).tz_localize(None).normalize()
        except Exception:
            dt=None
        pred_ret=clean_number(r.get("pred_ret"))
        actual_ret=clean_number(r.get("actual_ret"))
        low_ret=clean_number(r.get("low80"))
        high_ret=clean_number(r.get("high80"))
        start_price=None
        if dt is not None and price_col in prices.columns and dt in prices.index:
            start_price=clean_number(prices.at[dt,price_col])
        predicted_price=(start_price*(1+pred_ret)
                         if start_price is not None and pred_ret is not None else None)
        actual_price=(start_price*(1+actual_ret)
                      if start_price is not None and actual_ret is not None else None)
        low_price=(start_price*(1+low_ret)
                   if start_price is not None and low_ret is not None else None)
        high_price=(start_price*(1+high_ret)
                    if start_price is not None and high_ret is not None else None)
        points.append({
            "date":str(date_raw),
            "targetDate":str((dt+pd.Timedelta(days=horizon)).date()) if dt is not None else None,
            "probUp":clean_number(r.get("prob_up")) or 0,
            "predReturn":pred_ret or 0,
            "actualReturn":actual_ret,
            "startPrice":start_price,
            "predictedPrice":predicted_price,
            "actualPrice":actual_price,
            "low80Price":low_price,
            "high80Price":high_price,
        })
    last_date = points[-1]["date"] if points else None
    last_target = points[-1]["targetDate"] if points else None
    return {
        "coin": coin, "horizon": horizon, "points": points,
        "source": "local-oos-fallback" if LOCAL_MODE else "local-oos",
        "lastHistoryDate": last_date,
        "lastObservedTargetDate": last_target,
        "note": ("Online geschiedenis niet beschikbaar; lokale historische toetsresultaten getoond."
                 if LOCAL_MODE else "Lokale historische toetsresultaten."),
    }
