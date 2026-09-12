from __future__ import annotations
import math, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss,
    log_loss, mean_absolute_error, mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance

from config import Config

warnings.filterwarnings("ignore")

def _tree_n(cfg):
    return 600 if cfg.mode == "deep" else 250

def classifier_models(cfg: Config):
    n = _tree_n(cfg)
    models = {
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("m", RandomForestClassifier(
                n_estimators=n, max_depth=8, min_samples_leaf=15,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=11, n_jobs=-1
            ))
        ]),
        "hgb": Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=False)),
            ("m", HistGradientBoostingClassifier(
                learning_rate=0.04, max_iter=250 if cfg.mode=="deep" else 160,
                max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                random_state=23
            ))
        ]),
        "logit": Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
            ("m", LogisticRegression(C=0.2, max_iter=1500, class_weight="balanced"))
        ]),
    }
    if cfg.mode == "deep":
        models["extra"] = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("m", ExtraTreesClassifier(
                n_estimators=n, max_depth=9, min_samples_leaf=12,
                max_features="sqrt", class_weight="balanced",
                random_state=17, n_jobs=-1
            ))
        ])
        try:
            from xgboost import XGBClassifier
            models["xgb"] = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", XGBClassifier(
                    n_estimators=500, max_depth=4, learning_rate=0.025,
                    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                    eval_metric="logloss", random_state=29, n_jobs=-1
                ))
            ])
        except Exception:
            pass
    return models

def regressor_models(cfg: Config):
    n = _tree_n(cfg)
    models = {
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("m", RandomForestRegressor(
                n_estimators=n, max_depth=8, min_samples_leaf=15,
                max_features=0.7, random_state=31, n_jobs=-1
            ))
        ]),
        "hgb": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=250 if cfg.mode=="deep" else 160,
                max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                loss="huber", random_state=41
            ))
        ]),
        "ridge": Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
            ("m", Ridge(alpha=8.0))
        ]),
    }
    if cfg.mode == "deep":
        models["extra"] = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("m", ExtraTreesRegressor(
                n_estimators=n, max_depth=9, min_samples_leaf=12,
                max_features=0.7, random_state=37, n_jobs=-1
            ))
        ])
        try:
            from xgboost import XGBRegressor
            models["xgb"] = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", XGBRegressor(
                    n_estimators=500, max_depth=4, learning_rate=0.025,
                    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                    objective="reg:squarederror", random_state=43, n_jobs=-1
                ))
            ])
        except Exception:
            pass
    return models

def _valid_feature_subset(df, features, min_nonmissing=0.60, max_features=180):
    cols = []
    for c in features:
        if c not in df:
            continue
        ratio = df[c].notna().mean()
        if ratio >= min_nonmissing and df[c].nunique(dropna=True) > 1:
            cols.append(c)
    # Cap dimensionality using simple univariate variability/coverage score.
    if len(cols) > max_features:
        scores = []
        for c in cols:
            s = df[c]
            scores.append((c, s.notna().mean() * np.log1p(s.nunique(dropna=True))))
        cols = [c for c,_ in sorted(scores,key=lambda z:z[1],reverse=True)[:max_features]]
    return cols

def _model_weights_class(models, Xfit, yfit, Xval, yval):
    fitted={}
    losses={}
    for name,m in models.items():
        try:
            mm=clone(m).fit(Xfit,yfit)
            p=np.clip(mm.predict_proba(Xval)[:,1],1e-5,1-1e-5)
            losses[name]=brier_score_loss(yval,p)
            fitted[name]=mm
        except Exception:
            continue
    if not losses:
        return {},{}
    arr=np.array(list(losses.values()))
    # Exponential weighting; shifts by best so numerically stable.
    w=np.exp(-35*(arr-arr.min()))
    w=w/w.sum()
    return fitted,dict(zip(losses.keys(),w))

def _model_weights_reg(models, Xfit, yfit, Xval, yval):
    fitted={}
    losses={}
    for name,m in models.items():
        try:
            mm=clone(m).fit(Xfit,yfit)
            p=mm.predict(Xval)
            losses[name]=mean_absolute_error(yval,p)
            fitted[name]=mm
        except Exception:
            continue
    if not losses:
        return {},{}
    arr=np.array(list(losses.values()))
    scale=max(np.nanmedian(arr),1e-6)
    w=np.exp(-3*(arr-arr.min())/scale)
    w=w/w.sum()
    return fitted,dict(zip(losses.keys(),w))

def _ensemble_predict_class(fitted,weights,X):
    if not fitted: return np.full(len(X),np.nan)
    out=np.zeros(len(X)); denom=0
    for n,m in fitted.items():
        if n in weights:
            out += weights[n]*m.predict_proba(X)[:,1]
            denom += weights[n]
    return out/denom if denom else np.full(len(X),np.nan)

def _ensemble_predict_reg(fitted,weights,X):
    if not fitted: return np.full(len(X),np.nan)
    out=np.zeros(len(X)); denom=0
    for n,m in fitted.items():
        if n in weights:
            out += weights[n]*m.predict(X)
            denom += weights[n]
    return out/denom if denom else np.full(len(X),np.nan)

def walk_forward_expert(df, coin, horizon, features, cfg:Config, expert_name):
    yc=f"TARGET_{coin}_up_{horizon}"
    yr=f"TARGET_{coin}_ret_{horizon}"
    usable=df.dropna(subset=[yc,yr]).copy()
    selection_frame = usable.tail(1200) if "recent" in expert_name else usable
    features=_valid_feature_subset(selection_frame,features,min_nonmissing=0.40 if "recent" in expert_name else 0.65)
    if not features:
        return pd.DataFrame(),[]
    # Do not require complete rows; imputers handle source gaps. But require enough feature presence per row.
    row_present=usable[features].notna().mean(axis=1)
    usable=usable.loc[row_present>=0.50]
    if "recent" in expert_name:
        special=[c for c in features if c.startswith(("ETF_","CUSTOM_"))]
        if special:
            active=usable[special].notna().any(axis=1)
            if active.any():
                first_active=active[active].index.min()-pd.Timedelta(days=120)
                usable=usable.loc[usable.index>=first_active]
    if len(usable)<cfg.min_expert_rows:
        return pd.DataFrame(),features

    start_test=usable.index.min()+pd.DateOffset(years=cfg.min_train_years)
    test_dates=usable.index[usable.index>=start_test]
    rows=[]
    i=0
    cls_models=classifier_models(cfg)
    reg_models=regressor_models(cfg)

    while i<len(test_dates):
        block=test_dates[i:i+cfg.retrain_every_days]
        pred_start=block[0]
        # Purge/embargo: last training target must already be observable.
        label_cutoff=pred_start-pd.Timedelta(days=horizon)
        train=usable.loc[usable.index<=label_cutoff]
        test=usable.loc[block]
        if len(train)<cfg.min_expert_rows or test.empty:
            i+=cfg.retrain_every_days; continue

        val_days=min(cfg.validation_days,max(180,len(train)//5))
        fit=train.iloc[:-val_days]
        val=train.iloc[-val_days:]
        if len(fit)<500 or len(val)<cfg.min_validation_rows:
            i+=cfg.retrain_every_days; continue

        Xfit,Xval,Xtest=fit[features],val[features],test[features]
        yfitc,yvalc=fit[yc].astype(int),val[yc].astype(int)
        yfitr,yvalr=fit[yr],val[yr]

        fc,wc=_model_weights_class(cls_models,Xfit,yfitc,Xval,yvalc)
        fr,wr=_model_weights_reg(reg_models,Xfit,yfitr,Xval,yvalr)
        if not fc or not fr:
            i+=cfg.retrain_every_days; continue

        pc=_ensemble_predict_class(fc,wc,Xtest)
        pr=_ensemble_predict_reg(fr,wr,Xtest)

        # Empirical conformal-like residual band based ONLY on past validation residuals.
        val_pred=_ensemble_predict_reg(fr,wr,Xval)
        resid=yvalr.values-val_pred
        q10,q90=np.nanquantile(resid,[0.10,0.90])

        for dt,p,r,a,ar in zip(test.index,pc,pr,test[yc],test[yr]):
            rows.append({
                "date":dt,"coin":coin,"horizon":horizon,"expert":expert_name,
                "prob_up":float(p),"pred_ret":float(r),
                "low80":float(r+q10),"high80":float(r+q90),
                "actual_up":int(a),"actual_ret":float(ar),
                "n_features":len(features),
            })
        i+=cfg.retrain_every_days

    if not rows: return pd.DataFrame(),features
    return pd.DataFrame(rows).set_index("date").sort_index(),features

def metrics(bt:pd.DataFrame):
    if bt.empty: return {}
    y=bt.actual_up.astype(int)
    p=np.clip(bt.prob_up,1e-6,1-1e-6)
    pred=(p>=0.5).astype(int)
    out={
        "n":len(bt),
        "auc":roc_auc_score(y,p) if y.nunique()>1 else np.nan,
        "accuracy":accuracy_score(y,pred),
        "balanced_accuracy":balanced_accuracy_score(y,pred),
        "brier":brier_score_loss(y,p),
        "logloss":log_loss(y,p),
        "mae_return":mean_absolute_error(bt.actual_ret,bt.pred_ret),
        "rmse_return":mean_squared_error(bt.actual_ret,bt.pred_ret)**0.5,
        "coverage80":((bt.actual_ret>=bt.low80)&(bt.actual_ret<=bt.high80)).mean(),
    }
    return out

def meta_ensemble(expert_bts:Dict[str,pd.DataFrame], cfg:Config):
    all_dates=sorted(set().union(*[set(bt.index) for bt in expert_bts.values() if not bt.empty]))
    rows=[]
    for dt in all_dates:
        available=[]
        for name,bt in expert_bts.items():
            if bt.empty or dt not in bt.index: continue
            row=bt.loc[dt]
            if isinstance(row,pd.DataFrame): row=row.iloc[-1]
            hist=bt.loc[(bt.index<dt)&(bt.index>=dt-pd.Timedelta(days=cfg.recent_weight_window_days))]
            if len(hist)>=90:
                loss=np.mean((hist.prob_up-hist.actual_up)**2)
                mae=np.mean(np.abs(hist.pred_ret-hist.actual_ret))
            else:
                loss=np.nan; mae=np.nan
            available.append((name,row,loss,mae))
        if not available: continue
        losses=np.array([a[2] for a in available],dtype=float)
        if np.isfinite(losses).sum()<1:
            w=np.ones(len(available))/len(available)
        else:
            fill=np.nanmedian(losses[np.isfinite(losses)])
            losses=np.where(np.isfinite(losses),losses,fill+0.02)
            w=np.exp(-40*(losses-losses.min())); w=w/w.sum()
        prob=sum(wi*a[1].prob_up for wi,a in zip(w,available))
        pret=sum(wi*a[1].pred_ret for wi,a in zip(w,available))
        low=sum(wi*a[1].low80 for wi,a in zip(w,available))
        high=sum(wi*a[1].high80 for wi,a in zip(w,available))
        actual_up=int(available[0][1].actual_up)
        actual_ret=float(available[0][1].actual_ret)
        rows.append({
            "date":dt,"prob_up":prob,"pred_ret":pret,"low80":low,"high80":high,
            "actual_up":actual_up,"actual_ret":actual_ret,
            "experts_used":";".join(a[0] for a in available),
            "weights":";".join(f"{a[0]}={wi:.3f}" for wi,a in zip(w,available)),
        })
    return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()

def strategy_metrics(meta:pd.DataFrame, daily_price:pd.Series, cfg:Config):
    if meta.empty: return {}
    # Use each day's horizon probability as a daily risk signal, applied to next-day return.
    p=meta.prob_up.reindex(daily_price.index).ffill(limit=cfg.retrain_every_days)
    pos=pd.Series(0.0,index=daily_price.index)
    pos[p>=cfg.long_threshold]=0.5
    pos[p>=cfg.strong_long_threshold]=1.0
    pos[p<cfg.risk_off_threshold]=0.0
    next_ret=daily_price.pct_change().shift(-1)
    turn=pos.diff().abs().fillna(pos.iloc[0])
    strat=pos*next_ret-turn*cfg.transaction_cost
    z=pd.DataFrame({"strategy":strat,"buyhold":next_ret,"position":pos}).dropna()
    if len(z)<100:return {}
    eq=(1+z.strategy).cumprod()
    bh=(1+z.buyhold).cumprod()
    years=len(z)/365.25
    cagr=eq.iloc[-1]**(1/years)-1
    bhcagr=bh.iloc[-1]**(1/years)-1
    vol=z.strategy.std()*np.sqrt(365)
    sharpe=z.strategy.mean()/z.strategy.std()*np.sqrt(365) if z.strategy.std()>0 else np.nan
    dd=eq/eq.cummax()-1
    return {
        "n_days":len(z),"strategy_cagr":cagr,"buyhold_cagr":bhcagr,
        "strategy_vol":vol,"strategy_sharpe":sharpe,
        "max_drawdown":dd.min(),"avg_position":z.position.mean(),
        "turnover":turn.loc[z.index].sum(),
    }

def latest_expert_prediction(df,coin,horizon,features,cfg,expert_name):
    yc=f"TARGET_{coin}_up_{horizon}"
    yr=f"TARGET_{coin}_ret_{horizon}"
    usable=df.copy()
    sel=usable.dropna(subset=[yc,yr])
    selection_frame=sel.tail(1200) if "recent" in expert_name else sel
    features=_valid_feature_subset(selection_frame,features,
                                   min_nonmissing=0.40 if "recent" in expert_name else 0.65)
    if not features:return None
    latest=df[features].notna().mean(axis=1)
    latest_dates=latest[latest>=0.50].index
    if len(latest_dates)==0:return None
    dt=latest_dates.max()
    cutoff=dt-pd.Timedelta(days=horizon)
    train=df.loc[df.index<=cutoff].dropna(subset=[yc,yr])
    train=train.loc[train[features].notna().mean(axis=1)>=0.50]
    if len(train)<cfg.min_expert_rows:return None
    val_days=min(cfg.validation_days,max(180,len(train)//5))
    fit,val=train.iloc[:-val_days],train.iloc[-val_days:]
    if len(fit)<500 or len(val)<cfg.min_validation_rows:return None

    fc,wc=_model_weights_class(classifier_models(cfg),fit[features],fit[yc].astype(int),
                               val[features],val[yc].astype(int))
    fr,wr=_model_weights_reg(regressor_models(cfg),fit[features],fit[yr],
                             val[features],val[yr])
    if not fc or not fr:return None
    X=df.loc[[dt],features]
    # Out-of-distribution indicator: fraction of current feature values outside
    # the historical 1st–99th percentile range of the training sample.
    q01=train[features].quantile(0.01)
    q99=train[features].quantile(0.99)
    cur=X.iloc[0]
    comparable=cur.notna() & q01.notna() & q99.notna()
    ood_fraction=float((((cur<q01)|(cur>q99)) & comparable).sum()/max(1,comparable.sum()))
    p=float(_ensemble_predict_class(fc,wc,X)[0])
    r=float(_ensemble_predict_reg(fr,wr,X)[0])
    vr=_ensemble_predict_reg(fr,wr,val[features])
    resid=val[yr].values-vr
    q10,q90=np.nanquantile(resid,[.10,.90])
    return {"date":dt,"expert":expert_name,"prob_up":p,"pred_ret":r,
            "low80":r+q10,"high80":r+q90,"n_features":len(features),
            "ood_fraction":ood_fraction}

def latest_meta(latest_preds:List[dict], oos_bts:Dict[str,pd.DataFrame], cfg:Config):
    if not latest_preds:return None
    dt=max(p["date"] for p in latest_preds)
    usable=[p for p in latest_preds if p["date"]>=dt-pd.Timedelta(days=3)]
    losses=[]
    for p in usable:
        bt=oos_bts.get(p["expert"],pd.DataFrame())
        hist=bt.tail(cfg.recent_weight_window_days) if not bt.empty else bt
        loss=np.mean((hist.prob_up-hist.actual_up)**2) if len(hist)>=90 else np.nan
        losses.append(loss)
    arr=np.array(losses,dtype=float)
    if np.isfinite(arr).sum():
        fill=np.nanmedian(arr[np.isfinite(arr)])
        arr=np.where(np.isfinite(arr),arr,fill+0.02)
        w=np.exp(-40*(arr-arr.min())); w=w/w.sum()
    else:
        w=np.ones(len(usable))/len(usable)
    prob=sum(wi*p["prob_up"] for wi,p in zip(w,usable))
    disagreement=float(np.sqrt(sum(wi*(p["prob_up"]-prob)**2 for wi,p in zip(w,usable))))
    ood=float(sum(wi*p.get("ood_fraction",0.0) for wi,p in zip(w,usable)))
    # Heuristic confidence is descriptive, not a probability:
    # lower expert disagreement and lower OOD -> higher trust in model stability.
    confidence=float(np.clip(1.0 - 2.5*disagreement - 0.8*ood, 0, 1))
    return {
        "date":dt,
        "prob_up":prob,
        "pred_ret":sum(wi*p["pred_ret"] for wi,p in zip(w,usable)),
        "low80":sum(wi*p["low80"] for wi,p in zip(w,usable)),
        "high80":sum(wi*p["high80"] for wi,p in zip(w,usable)),
        "weights":{p["expert"]:float(wi) for wi,p in zip(w,usable)},
        "expert_disagreement":disagreement,
        "ood_fraction":ood,
        "confidence_score":confidence,
    }
