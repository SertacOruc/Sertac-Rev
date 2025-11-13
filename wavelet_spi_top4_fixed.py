#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOP-4 — Wavelet Ablation — FAST & STABLE — Extended (Single Script)
===================================================================
Bu betik 1. sürümün geliştirilmiş tek-dosya versiyonudur.
"""

# -------------------- ENV (importlardan ÖNCE) --------------------
import os, sys
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import subprocess, importlib, warnings, re, unicodedata as ucd, time, json
warnings.filterwarnings("ignore")

# --------- Modül güvence helper'ı ----------
def _ensure(mod, pip_name=None, asname=None, quiet=True):
    try:
        m = importlib.import_module(mod)
    except ImportError:
        try:
            cmd=[sys.executable,"-m","pip","install"]
            if quiet: cmd.append("--quiet")
            cmd.append(pip_name or mod)
            subprocess.check_call(cmd)
            m = importlib.import_module(mod)
        except Exception:
            return None
    if asname: globals()[asname]=m
    else: globals()[mod]=m
    return m

# Temel modüller
for mod,pipn,asn in [
    ("numpy","numpy","np"),
    ("pandas","pandas","pd"),
    ("matplotlib","matplotlib",None),
    ("matplotlib.pyplot","matplotlib","plt"),
    ("matplotlib.dates","matplotlib","mdates"),
    ("mpl_toolkits.mplot3d","matplotlib","mplot3d"),
    ("sklearn","scikit-learn",None),
    ("pywt","PyWavelets","pywt"),
    ("scipy","scipy",None),
    ("joblib","joblib>=1.3.2","joblib"),
]:
    _ensure(mod,pipn,asn)

def _purge_tk_modules():
    for _m in ['tkinter','_tkinter','tkinter.ttk','tkinter.messagebox']:
        if _m in sys.modules:
            try: del sys.modules[_m]
            except Exception: pass

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from joblib import parallel_backend
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.inspection import permutation_importance
try:
    from sklearn.inspection import PartialDependenceDisplay as _PDPDisplay
    _HAS_PDP = True
except Exception:
    _HAS_PDP = False
from scipy.stats import gamma as sp_gamma, norm as sp_norm, gaussian_kde as sp_kde, probplot as sp_probplot, skew, kurtosis, shapiro

plt.ioff()

# ============================================================
# KULLANICI AYARLARI - BURAYA KENDİ YOLLARINIZI YAZIN
# ============================================================
# Windows yolları için forward slash (/) veya çift backslash (\\) kullanın
INPUT_XLSX  = r"C:/Users/ser_o/OneDrive/Documents/veri.xlsx"
OUTDIR_BASE = r"C:/Users/ser_o/OneDrive/Documents/sonuclar"

# Çok-ölçekli ve çok-adımlı hedef
SPI_SCALES = [12]      # örn: [3,6,12]
LEADS      = [0]       # örn: [0,1,3,6,12] — 0: eşzamanlı

# Seçim kriteri ve Top-K
SELECT_BY = "KGE"        # "KGE" veya "RMSE"
TOP_K     = 4

# Çalışma modu
ID_COL = "Istasyon_No"
RUN_ALL_STATIONS  = True    # False = tek istasyon
STATION_ID        = None    # Tek istasyon modunda ID (None = ilk istasyon)
PLOTS_FOR_ALL     = True    # Tüm istasyonlar için plot
MAKE_PLOTS_SINGLE = True    # Tek istasyon için plot

# Hız/kalite ayarları
RANDOM_STATE = 42
FAST_MODE = False
CV_SPLITS = 3 if FAST_MODE else 5
N_ITER    = 6 if FAST_MODE else 14
CV_GAP    = 36
MIN_TRAIN = 120

# Variant aileleri
ENABLE_WF  = True
ENABLE_WD  = True
ENABLE_WPD = True

# Modeller
MODELS_ON = {
    "HGBR": True,
    "SVR":  True,
    "RF":   True,
    "ET":   True,
    "GBR":  True,
    "RIDGE":True,
    "KNN":  True,
    "LSVR": True,
}

# Wavelet listeleri
WAVELETS_DB  = ["db3","db4"] if FAST_MODE else ["db2","db3","db4","db5","db6","db8","db10"]
WAVELETS_SYM = ["sym4"] if FAST_MODE else ["sym4","sym5"]
WAVELETS_COI = [] if FAST_MODE else ["coif3","coif4"]

# Gecikme yapıları
MAX_LAG = 12
LAG_STRUCTURES_FAST = {
    "M03":[1,2,12],
    "M06":[1,2,3,6,12],
    "M08":[1,2,3,4,5,6,11,12],
}
LAG_STRUCTURES_FULL = {
    "M01":[1],
    "M02":[1,2],
    "M03":[1,2,12],
    "M04":[1,2,11,12],
    "M05":[1,2,3,11,12],
    "M06":[1,2,3,6,12],
    "M07":[1,3,6,9,12],
    "M08":[1,2,3,4,5,6,11,12],
}
LAG_STRUCTURES = LAG_STRUCTURES_FAST if FAST_MODE else LAG_STRUCTURES_FULL

# Opsiyonel Tanılama Bayrakları
HEADLESS_HARDENING       = True
DATE_ALIGN_STRATEGY      = "intersection"

# SPI kalibrasyon modu
SPI_CALIBRATION_MODE   = "all"
SPI_CALIB_TRAIN_RATIO  = 0.70
SPI_CLIM_START         = "1981-01-01"
SPI_CLIM_END           = "2010-12-31"

# Özellik seçimi
ENABLE_FEATURE_SELECTION = False
FEATURE_SELECTION_MODE   = "variance"
VARIANCE_THRESHOLD       = 0.0
SFM_MAX_FEATURES         = None
SFM_THRESHOLD            = "median"

# EK GÖRSEL/ÇIKTI MODÜLLERİ
ENABLE_MONTH_SKILL_HEATMAP = False
ENABLE_PERM_IMPORTANCE     = False
ENABLE_PDP_ICE             = False
ENABLE_QUALITY_REPORT      = False

# ---------------- Yayın stili ----------------
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False
})
if HEADLESS_HARDENING:
    _purge_tk_modules()

def hide_spines(ax):
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass

# ----------------- Yardımcılar -----------------
def read_excel_safely(path, sheet_name=0):
    last=None
    if _ensure("openpyxl","openpyxl"):
        try: return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        except Exception as e: last=e
    if _ensure("calamine","pandas-calamine"):
        try: return pd.read_excel(path, sheet_name=sheet_name, engine="calamine")
        except Exception as e: last=e
    if str(path).lower().endswith(".xls") and _ensure("xlrd","xlrd<2.0"):
        try: return pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
        except Exception as e: last=e
    raise RuntimeError(f"Excel okunamadı: {last}")

def _norm_text(s:str)->str:
    s=str(s).replace("\u0130","I").replace("\u0131","i")
    s=ucd.normalize("NFD",s)
    s="".join(ch for ch in s if ucd.category(ch)!="Mn")
    s=re.sub(r"[^0-9A-Za-z_]+","_",s).strip("_")
    return s

def load_by_station_id(path, id_col=ID_COL):
    df=read_excel_safely(path)
    if id_col not in df.columns:
        cols_norm={_norm_text(c).upper():c for c in df.columns}
        cand=None
        for k in ["ISTASYON_NO","ISTASYONNO","STATION_NO","NO","IST_NO"]:
            if k in cols_norm: cand=cols_norm[k]; break
        if cand is None:
            raise ValueError(f"'{id_col}' sütunu bulunamadı.")
        id_col=cand

    cols={_norm_text(c).upper():c for c in df.columns}
    def pick(*cands):
        for k in cands:
            if k in cols: return cols[k]
        return None
    col_name = pick("ISTASYON_ADI","STATION_NAME","ISTASYON")
    col_yil  = pick("YIL","YEAR")
    col_ay   = pick("AY","MONTH","AY_NO","MONTH_NO")

    val_col=None
    for u,c in cols.items():
        if ("YAG" in u or "RAIN" in u or "PRECIP" in u) and ("AYLIK" in u or "TOPLAM" in u or "TOTAL" in u or "MONTH" in u or "MM" in u):
            val_col=c; break
    if val_col is None:
        for u,c in cols.items():
            if "MANUEL" in u and ("YAG" in u or "RAIN" in u or "PRECIP" in u):
                val_col=c; break
    if val_col is None: raise ValueError("Aylık yağış kolonu bulunamadı.")
    if col_yil is None or col_ay is None: raise ValueError("YIL/AY kolonları gerekli.")

    stations={}
    for sid, sub in df.groupby(id_col):
        sub=sub.copy()
        year = pd.to_numeric(sub[col_yil].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
        month= pd.to_numeric(sub[col_ay].astype(str).str.extract(r"(\d{1,2})")[0], errors="coerce")
        sub=sub.assign(_year=year,_month=month).dropna(subset=["_year","_month"])
        sub=sub[(sub["_month"]>=1)&(sub["_month"]<=12)&(sub["_year"].between(1800,2100))].copy()
        sub["_year"]=sub["_year"].astype(int); sub["_month"]=sub["_month"].astype(int)
        sub["date"]=pd.to_datetime(sub["_year"].astype(str)+"-"+sub["_month"].astype(str).str.zfill(2)+"-01", errors="coerce")
        sub=sub.dropna(subset=["date"]).sort_values("date")

        ts=sub[["date",val_col]].rename(columns={val_col:"precip"}).copy()
        ts["precip"]=pd.to_numeric(ts["precip"],errors="coerce").clip(lower=0.0)
        full=pd.date_range(ts["date"].min(), ts["date"].max(), freq="MS")
        ts=ts.set_index("date").reindex(full); ts.index.name="date"; ts=ts.reset_index()

        label = f"{sid}" if col_name is None else f"{sid} | {str(sub[col_name].iloc[0])}"
        stations[str(sid)]={"label":label, "df":ts}
    return stations, id_col

# -------------- SPI --------------
def _compute_global_calib_end(s_valid_dates, ratio=0.7):
    if len(s_valid_dates)==0:
        return None
    k = max(1, int(np.floor(len(s_valid_dates)*ratio)))
    return pd.to_datetime(s_valid_dates.iloc[k-1])

def compute_spi(df, value_col="precip", date_col="date", scale=12,
                calib_mode="all", calib_train_ratio=0.7,
                calib_start=None, calib_end=None, min_pos_samples=6):
    s=df[[date_col,value_col]].copy()
    s[date_col]=pd.to_datetime(s[date_col])
    s[value_col]=pd.to_numeric(s[value_col],errors="coerce").clip(lower=0.0)
    s=s.dropna(subset=[value_col]).sort_values(date_col)
    s["month"]=s[date_col].dt.month
    s["agg"]=s[value_col].rolling(window=scale, min_periods=scale).sum()

    if calib_mode=="fixed_range":
        cal_start = pd.to_datetime(calib_start) if calib_start else None
        cal_end   = pd.to_datetime(calib_end) if calib_end else None
    elif calib_mode=="train_ratio":
        valid = s.loc[s["agg"].notna(), date_col]
        cal_end = _compute_global_calib_end(valid.reset_index(drop=True), ratio=calib_train_ratio)
        cal_start = None
    else:
        cal_start = cal_end = None

    spi=pd.Series(index=s.index, dtype=float)
    for m in range(1,13):
        idx  = (s["month"]==m) & s["agg"].notna()
        vals = s.loc[idx, "agg"].values
        if len(vals)<12:
            continue

        if calib_mode=="fixed_range":
            idx_cal = idx & ( (cal_start is None) | (s[date_col]>=cal_start) ) & ( (cal_end is None) | (s[date_col]<=cal_end) )
        elif calib_mode=="train_ratio":
            idx_cal = idx & ( (cal_end is None) | (s[date_col]<=cal_end) )
        else:
            idx_cal = idx

        pos_cal = s.loc[idx_cal, "agg"].values
        pos_cal = pos_cal[pos_cal>0.0]
        if len(pos_cal) < min_pos_samples:
            pos_cal = vals[vals>0.0]

        H = (vals==0).sum()/len(vals)
        pos = vals[vals>0.0]
        if len(pos) < min_pos_samples:
            continue
        try:
            a,loc,b = sp_gamma.fit(pos_cal, floc=0.0)
        except Exception:
            continue
        G = sp_gamma.cdf(vals,a,loc=0.0,scale=b) if a>0 and b>0 else np.zeros_like(vals)
        p = np.clip(H+(1.0-H)*G, 1e-6, 1-1e-6)
        spi.loc[idx]=sp_norm.ppf(p)

    out=s[[date_col]].copy()
    out[f"spi_{scale}"]=spi.values
    return out

# -------------- Özellik üretimi --------------
def add_lags_max(df, ycol="y", prefix="lag", max_lag=MAX_LAG):
    for L in range(1, max_lag+1):
        df[f"{prefix}{L}"]=df[ycol].shift(L)
    return df

def dwt_window_features(series, wavelet="db4", level=3, window=36):
    import pywt
    s=pd.Series(series).astype(float).values
    n=len(s)
    _=pywt.wavedec(np.zeros(window), wavelet, level=level)
    names=[f"A{level}"]+[f"D{k}" for k in range(level,0,-1)]
    stats=["E","M","S","X","rE"]
    feats={f"WF_{wavelet}_{nm}_{st}":[np.nan]*n for nm in names for st in stats}
    for i in range(window,n):
        arr=s[i-window:i]
        if np.isnan(arr).any(): continue
        coeffs=pywt.wavedec(arr,wavelet,level=level)
        energies=[]
        for nm,c in zip(names, coeffs):
            c=np.asarray(c); e=float(np.sum(c**2)); energies.append(e)
            feats[f"WF_{wavelet}_{nm}_E"][i]=e
            feats[f"WF_{wavelet}_{nm}_M"][i]=float(np.mean(np.abs(c)))
            feats[f"WF_{wavelet}_{nm}_S"][i]=float(np.std(c))
            feats[f"WF_{wavelet}_{nm}_X"][i]=float(np.max(np.abs(c)))
        tot=float(np.sum(energies))+1e-12
        for nm,e in zip(names, energies):
            feats[f"WF_{wavelet}_{nm}_rE"][i]=e/tot
    return pd.DataFrame(feats)

def dwt_causal_denoise(series, wavelet="db4", level=3, window=36, mode="soft"):
    import pywt
    s=pd.Series(series).astype(float).values
    n=len(s); y_dn=np.full(n, np.nan)
    for i in range(window,n):
        arr=s[i-window:i]
        if np.isnan(arr).any(): continue
        coeffs=pywt.wavedec(arr,wavelet,level=level)
        d1=coeffs[-1]
        sigma=np.median(np.abs(d1-np.median(d1)))/0.6745 + 1e-12
        thr=sigma*np.sqrt(2*np.log(len(arr)))
        new_coeffs=[coeffs[0]]+[pywt.threshold(c, value=thr, mode=mode) for c in coeffs[1:]]
        rec=pywt.waverec(new_coeffs,wavelet)
        y_dn[i]=rec[-1]
    return pd.Series(y_dn)

def wpd_window_features(series, wavelet="db4", level=3, window=36):
    import pywt
    s=pd.Series(series).astype(float).values
    n=len(s)
    feats={}
    dummy=np.zeros(window)
    wp=pywt.WaveletPacket(data=dummy, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes=[node.path for node in wp.get_level(level, order='natural')]
    for p in nodes:
        feats[f"WPD_{wavelet}_{p}_E"]=[np.nan]*n
        feats[f"WPD_{wavelet}_{p}_rE"]=[np.nan]*n
    for i in range(window,n):
        arr=s[i-window:i]
        if np.isnan(arr).any(): continue
        wp=pywt.WaveletPacket(data=arr, wavelet=wavelet, mode='symmetric', maxlevel=level)
        lvl=wp.get_level(level, order='natural')
        energies=[float(np.sum(np.asarray(node.data)**2)) for node in lvl]
        tot=float(np.sum(energies))+1e-12
        for e,node in zip(energies,lvl):
            feats[f"WPD_{wavelet}_{node.path}_E"][i]=e
            feats[f"WPD_{wavelet}_{node.path}_rE"][i]=e/tot
    return pd.DataFrame(feats)

def build_variant_frames(df_spi, target_col, align_strategy="intersection"):
    base=df_spi.rename(columns={target_col:"y"}).copy()
    base=add_lags_max(base,"y","lag",MAX_LAG)
    frames={}

    y_in=base["y"].interpolate(limit_direction="forward")
    window, level = 36, 3

    frames["NW"]=base.copy()

    if ENABLE_WF:
        WF_WAVS = WAVELETS_DB + WAVELETS_SYM + WAVELETS_COI
        for w in WF_WAVS:
            wf=dwt_window_features(y_in.shift(1).fillna(y_in.median()), wavelet=w, level=level, window=window)
            frames[f"WF-{w}"]=pd.concat([base, wf], axis=1)

    if ENABLE_WD:
        WD_WAVS = (WAVELETS_DB + WAVELETS_SYM + WAVELETS_COI) or ["db4"]
        for w in WD_WAVS:
            ydn=dwt_causal_denoise(y_in, wavelet=w, level=level, window=window, mode="soft")
            dn=pd.DataFrame({"y_dn":ydn})
            dn=add_lags_max(dn,"y_dn","dnlag",MAX_LAG)
            frames[f"WD-{w}"]=pd.concat([base[["date","y"]], dn.drop(columns=["y_dn"])], axis=1)

    if ENABLE_WPD:
        WPD_WAVS = (WAVELETS_DB[:2]+["sym4"]) if FAST_MODE else (WAVELETS_DB + WAVELETS_SYM)
        for w in WPD_WAVS:
            wp=wpd_window_features(y_in.shift(1).fillna(y_in.median()), wavelet=w, level=level, window=window)
            frames[f"WPD-{w}"]=pd.concat([base, wp], axis=1)

    # Tarih hizalama
    if align_strategy=="intersection":
        date_sets=[]
        for _,df in frames.items():
            req=[c for c in df.columns if c not in ["date","y"]]
            mask=df[["y"]+req].notna().all(axis=1)
            date_sets.append(set(pd.to_datetime(df.loc[mask,"date"]).astype("datetime64[ns]")))
        common_dates=sorted(set.intersection(*date_sets)) if date_sets else []
        for k,df in frames.items():
            req=[c for c in df.columns if c not in ["date","y"]]
            df["date"]=pd.to_datetime(df["date"])
            df=df[df["date"].isin(common_dates)].copy()
            df=df[["date","y"]+req].sort_values("date").reset_index(drop=True)
            frames[k]=df
    else:  # 'min_start'
        first_dates=[]
        for k,df in frames.items():
            req=[c for c in df.columns if c not in ["date","y"]]
            mask=df[["y"]+req].notna().all(axis=1)
            d= pd.to_datetime(df.loc[mask,"date"])
            if len(d)>0: first_dates.append(d.min())
        global_start= max(first_dates) if first_dates else None
        for k,df in frames.items():
            req=[c for c in df.columns if c not in ["date","y"]]
            df["date"]=pd.to_datetime(df["date"])
            if global_start is not None:
                df=df[df["date"]>=global_start].copy()
            df=df[["date","y"]+req].dropna().sort_values("date").reset_index(drop=True)
            frames[k]=df

    return frames

# -------------- Metrikler --------------
def rmse(y_true,y_pred): return float(np.sqrt(mean_squared_error(y_true,y_pred)))
def nse(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    denom = np.sum((y - np.mean(y))**2)
    return 1.0 - np.sum((yhat - y)**2) / denom if denom>0 else np.nan
def willmott_d(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    num = np.sum((yhat - y)**2)
    den = np.sum((np.abs(yhat - np.mean(y)) + np.abs(y - np.mean(y)))**2)
    return 1.0 - num/den if den>0 else np.nan
def kge_2009(y_true,y_pred, eps=1e-8):
    mu_o,mu_g=np.mean(y_true),np.mean(y_pred)
    so,sg=np.std(y_true,ddof=0),np.std(y_pred,ddof=0)
    r=np.corrcoef(y_true,y_pred)[0,1]
    if not np.isfinite([mu_o,mu_g,so,sg,r]).all(): return float("nan")
    if abs(mu_o)<eps or abs(mu_g)<eps or so==0 or sg==0:
        return float(r)
    beta=mu_g/mu_o; cvo=so/mu_o; cvg=sg/mu_g; gamma=cvg/cvo if cvo!=0 else np.nan
    return float(1.0 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2))
def metrics_all(y_true, y_pred):
    return dict(R=float(np.corrcoef(y_true,y_pred)[0,1]),
                R2=float(r2_score(y_true,y_pred)),
                MAE=float(mean_absolute_error(y_true,y_pred)),
                RMSE=rmse(y_true,y_pred),
                KGE=kge_2009(y_true,y_pred),
                NSE=nse(y_true,y_pred),
                WD =willmott_d(y_true,y_pred))
RMSE_SCORER = make_scorer(lambda yt, yp: -rmse(yt, yp))

# ----------- Özellik Seçimi Yardımcısı -----------
def _make_feat_selector():
    if not ENABLE_FEATURE_SELECTION:
        return None
    if FEATURE_SELECTION_MODE=="variance":
        return VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    elif FEATURE_SELECTION_MODE=="from_model":
        base = HistGradientBoostingRegressor(random_state=RANDOM_STATE, early_stopping=False)
        return SelectFromModel(estimator=base, threshold=SFM_THRESHOLD, max_features=SFM_MAX_FEATURES)
    else:
        return None

# -------------- Modeller ve arama uzayı --------------
def model_space(name):
    use_scaler = name in {"SVR","RIDGE","KNN","LSVR"}
    feat_sel   = _make_feat_selector()

    if name=="SVR":
        base = SVR(kernel="rbf")
        space={"model__C":np.logspace(0,3,10),
               "model__gamma":np.logspace(-4,-1,8),
               "model__epsilon":np.linspace(0.01,0.2,5)}
    elif name=="RF":
        base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
        space={"model__n_estimators":np.linspace(200,400,4,dtype=int),
               "model__max_depth":[None,6,10],
               "model__min_samples_split":[2,5],
               "model__min_samples_leaf":[1,2,4],
               "model__max_features":["sqrt",0.7]}
    elif name=="ET":
        base = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1)
        space={"model__n_estimators":np.linspace(200,400,4,dtype=int),
               "model__max_depth":[None,6,10],
               "model__min_samples_split":[2,5],
               "model__min_samples_leaf":[1,2,4],
               "model__max_features":["sqrt",0.7]}
    elif name=="GBR":
        base = GradientBoostingRegressor(random_state=RANDOM_STATE)
        space={"model__n_estimators":np.linspace(150,300,4,dtype=int),
               "model__learning_rate":np.logspace(-3,-0.6,6),
               "model__max_depth":[2,3,4],
               "model__min_samples_leaf":[1,2,4],
               "model__subsample":[0.7,1.0]}
    elif name=="HGBR":
        base = HistGradientBoostingRegressor(random_state=RANDOM_STATE, early_stopping=False)
        space={"model__learning_rate":np.logspace(-3,-0.3,8),
               "model__max_depth":[None,3,5],
               "model__max_leaf_nodes":[31,63,127],
               "model__min_samples_leaf":[1,5,10],
               "model__l2_regularization":np.logspace(-4,1,6)}
    elif name=="RIDGE":
        base = Ridge()
        space={"model__alpha":np.logspace(-3,3,12),
               "model__fit_intercept":[True,False]}
    elif name=="KNN":
        base = KNeighborsRegressor()
        space={"model__n_neighbors":np.arange(2,21),
               "model__weights":["uniform","distance"],
               "model__p":[1,2]}
    elif name=="LSVR":
        base = LinearSVR(random_state=RANDOM_STATE,max_iter=10000)
        space={"model__C":np.logspace(-3,2,10),
               "model__epsilon":np.linspace(0.0,0.2,5),
               "model__loss":["epsilon_insensitive","squared_epsilon_insensitive"]}
    else:
        raise ValueError(name)

    steps=[]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if feat_sel is not None:
        steps.append(("feat_sel", feat_sel))
    steps.append(("model", base))
    est = Pipeline(steps)
    return est, space

CANDIDATES = [m for m,on in MODELS_ON.items() if on]

def fit_cv(est, space, X, y, gap=CV_GAP):
    try:
        cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=gap)
    except TypeError:
        cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    rs = RandomizedSearchCV(est, space, n_iter=N_ITER, random_state=RANDOM_STATE,
                            cv=cv, scoring=RMSE_SCORER, n_jobs=-1, verbose=0)
    with parallel_backend("loky", inner_max_num_threads=1):
        rs.fit(X, y)
    return rs.best_estimator_, -rs.best_score_, rs.best_params_

# -------------- Çizim yardımcıları --------------
def year_axis(ax, step=5):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

def set_xtick_rotation(ax, rot=45, ha='right'):
    for t in ax.get_xticklabels():
        try: t.set_rotation(rot)
        except Exception: pass
        try: t.set_horizontalalignment(ha)
        except Exception: pass

def _naive_vec(y, m=1):
    y=np.asarray(y,dtype=float)
    yhat=np.full_like(y,np.nan,dtype=float)
    if m<len(y): yhat[m:]=y[:-m]
    return yhat

def rolling_rmse(y, yhat, win=24):
    e=(yhat-y)**2
    return pd.Series(e).rolling(win).mean().apply(np.sqrt).values

# ----------- Lead uygulama -----------
def apply_lead_to_variants(frames, lead):
    if lead==0: return frames
    out={}
    for k,df in frames.items():
        d = df.copy()
        d["y_lead"] = d["y"].shift(-lead)
        d = d.drop(columns=["y"]).rename(columns={"y_lead":"y"})
        out[k] = d
    return out

# -------------- Basit değerlendirme çekirdeği --------------
def evaluate_station_simple(label, df_date_precip, spi_scale=12, lead=0):
    """Basitleştirilmiş versiyon - sadece metrik hesabı, plot yok"""
    tag_base = f"{_norm_text(label)[:40]}_SPI{spi_scale}_LEAD{lead}"

    print(f"\n{'='*70}")
    print(f"İŞLENİYOR: {label}")
    print(f"SPI Ölçeği: {spi_scale} | Lead: {lead}")
    print(f"{'='*70}")

    # SPI hesapla
    spi=compute_spi(
        df_date_precip, value_col="precip", date_col="date", scale=spi_scale,
        calib_mode=SPI_CALIBRATION_MODE, calib_train_ratio=SPI_CALIB_TRAIN_RATIO,
        calib_start=SPI_CLIM_START, calib_end=SPI_CLIM_END
    )
    data=df_date_precip[["date"]].merge(spi, on="date", how="left").rename(columns={f"spi_{spi_scale}":"y"})

    # Varyantlar
    variants=build_variant_frames(data[["date","y"]].copy(), target_col="y", align_strategy=DATE_ALIGN_STRATEGY)
    variants=apply_lead_to_variants(variants, lead=lead)

    any_key=list(variants.keys())[0]
    vdf0=variants[any_key][["date","y"]].dropna().reset_index(drop=True)
    y0=vdf0["y"].values; d0=vdf0["date"].values
    split0=int(0.7*len(vdf0))

    results=[]

    # Naive baseline
    if lead==0 and split0>=1 and len(vdf0)-split0>=1:
        for nm,m in [("Naive-1",1),("Naive-12",12)]:
            yhat=_naive_vec(y0,m=m)[split0:]; yte=y0[split0:]
            mask=np.isfinite(yhat)&np.isfinite(yte)
            if mask.sum()>=3:
                mets=metrics_all(yte[mask], yhat[mask])
                base_label=f"{nm} | NW | BASE"
                results.append({"Label":base_label,"Model":nm,"Variant":"NW","Struct":"BASE","CV_RMSE":np.nan,**mets})
                print(f"  {base_label:>22s}  RMSE={mets['RMSE']:.3f}  KGE={mets['KGE']:.3f}  R={mets['R']:.3f}")

    # Varyant+Yapı+Model döngüsü
    for vname, vdf in variants.items():
        base_cols=[c for c in vdf.columns if c not in ["date","y"] and not c.startswith("lag") and not c.startswith("dnlag")]
        for mcode, m_lags in LAG_STRUCTURES.items():
            lag_cols = [f"dnlag{L}" for L in m_lags] if vname.startswith("WD") else [f"lag{L}" for L in m_lags]
            feat_cols=lag_cols + base_cols
            df_use=vdf[["date","y"]+feat_cols].dropna()
            if len(df_use)<max(MIN_TRAIN, 3*MAX_LAG):
                continue
            X=df_use[feat_cols].values; y=df_use["y"].values
            split_local=int(0.7*len(df_use))
            Xtr,Xte=X[:split_local],X[split_local:]; ytr,yte=y[:split_local],y[split_local:]
            for mname in CANDIDATES:
                est,space=model_space(mname)
                best,cv_rmse,params=fit_cv(est,space,Xtr,ytr,gap=CV_GAP)
                yhat=best.predict(Xte)
                mets=metrics_all(yte,yhat)
                full_label=f"{mname} | {vname} | {mcode}"
                results.append({"Label":full_label,"Model":mname,"Variant":vname,"Struct":mcode,"CV_RMSE":cv_rmse,**mets})
                print(f"  {full_label:>22s}  RMSE={mets['RMSE']:.3f}  KGE={mets['KGE']:.3f}  R={mets['R']:.3f}")

    res=pd.DataFrame(results)
    if res.empty:
        print(f"[UYARI] {label} için geçerli sonuç üretilemedi")
        return None

    key="KGE" if SELECT_BY.upper()=="KGE" else "RMSE"
    asc=False if key=="KGE" else True
    topk=res.sort_values(key, ascending=asc).head(TOP_K)

    # Çıktıları kaydet
    os.makedirs(OUTDIR, exist_ok=True)
    res.to_csv(os.path.join(OUTDIR,f"{tag_base}_metrics_all.csv"), index=False)
    topk.to_csv(os.path.join(OUTDIR,f"{tag_base}_top{TOP_K}.csv"), index=False)

    print(f"\n✓ En iyi {TOP_K} sonuç kaydedildi")
    print(topk[["Label","RMSE","KGE","R"]].to_string(index=False))

    return {"station": label, "scale": spi_scale, "lead": lead, "topk": topk.copy()}

# -------------- Ana akış --------------
def main():
    global OUTDIR

    print("\n" + "="*70)
    print("WAVELET-SPI TAHMIN SİSTEMİ")
    print("="*70)
    print(f"Girdi dosyası: {INPUT_XLSX}")
    print(f"Çıktı klasörü: {OUTDIR_BASE}")
    print(f"SPI ölçekleri: {SPI_SCALES}")
    print(f"Lead değerleri: {LEADS}")
    print(f"Hızlı mod: {FAST_MODE}")
    print("="*70 + "\n")

    # Klasörleri oluştur
    os.makedirs(OUTDIR_BASE, exist_ok=True)
    OUTDIR = os.path.join(OUTDIR_BASE, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(OUTDIR, exist_ok=True)

    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {INPUT_XLSX}")

    # İstasyonları yükle
    stations_map, real_id_col = load_by_station_id(INPUT_XLSX, ID_COL)
    id_list=list(stations_map.keys())
    print(f"✓ {len(id_list)} istasyon yüklendi (ID sütunu: {real_id_col})\n")

    if RUN_ALL_STATIONS:
        summaries=[]
        for sid in id_list:
            st=stations_map[sid]
            for sc in SPI_SCALES:
                for ld in LEADS:
                    summ=evaluate_station_simple(st["label"], st["df"], spi_scale=sc, lead=ld)
                    if summ:
                        summaries.append(summ)

        # Toplu özet
        if summaries:
            rows=[]
            for s in summaries:
                best=s["topk"].iloc[0]
                rows.append({
                    "Station": s["station"],
                    "SPI": s["scale"],
                    "Lead": s["lead"],
                    "Best_Model": best["Model"],
                    "Best_Variant": best["Variant"],
                    "RMSE": best["RMSE"],
                    "KGE": best["KGE"],
                    "R": best["R"]
                })
            summary_df=pd.DataFrame(rows)
            summary_df.to_csv(os.path.join(OUTDIR,"ALL_stations_summary.csv"), index=False)
            print("\n" + "="*70)
            print("TÜM İSTASYONLAR ÖZET")
            print("="*70)
            print(summary_df.to_string(index=False))
    else:
        # Tek istasyon
        if STATION_ID is None:
            sid=id_list[0]
            print(f"[BİLGİ] STATION_ID belirtilmedi, ilk istasyon kullanılıyor: {sid}\n")
        else:
            sid=str(STATION_ID)
            if sid not in stations_map:
                print(f"[UYARI] STATION_ID={STATION_ID} bulunamadı, ilk istasyon kullanılacak.\n")
                sid=id_list[0]
        st=stations_map[sid]
        for sc in SPI_SCALES:
            for ld in LEADS:
                evaluate_station_simple(st["label"], st["df"], spi_scale=sc, lead=ld)

    print("\n" + "="*70)
    print("✓ ANALİZ TAMAMLANDI")
    print(f"Sonuçlar: {OUTDIR}")
    print("="*70 + "\n")

# ----------------- Çalıştırma -----------------
if __name__=="__main__":
    try:
        OUTDIR=""
        main()
    except KeyboardInterrupt:
        print("\n\n[İPTAL] Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n[HATA] {e}")
        import traceback; traceback.print_exc()
