#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOP-4 — Wavelet Ablation (SPI-12) — FAST & STABLE
=================================================
- FAST_MODE varsayılan: hızlı HPO (CV=3, n_iter=6), yalnızca WF aktif, modeller: HGBR+SVR
- WD/WPD, diğer modeller kapalı ama tek satırla açılabilir.
- Agg backend (GUI yok) → Tkinter hataları yok, sadece dosyaya kaydeder.
- Tek istasyon modunda STATION_ID boşsa ilk istasyonu otomatik seçer.

Çıktılar: outputs/ klasörüne PNG ve CSV dosyaları (yayın kalitesi dpi=300).

Yazan: AI Asistan (patched sürüm)
"""

# -------------------- ENV (importlardan ÖNCE) --------------------
import os, sys
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
# Başsız çizim (Tkinter uyarılarını engeller)
os.environ.setdefault("MPLBACKEND", "Agg")
# Aşırı paralelliği frenle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import subprocess, importlib, warnings, re, unicodedata as ucd
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
from scipy.stats import gamma as sp_gamma, norm as sp_norm, gaussian_kde as sp_kde, probplot as sp_probplot
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

plt.ioff()  # etkileşimi kapat (tamamen dosyaya yaz)

# ---------------- Kullanıcı ayarları (FAST) ----------------
OUTDIR = "outputs"
INPUT_XLSX = os.path.join(OUTDIR, "Stations_Thrace_Turker.xlsx")  # <-- dosya yolunu kendine göre ver
SPI_SCALE = 12
SELECT_BY = "KGE"        # "KGE" veya "RMSE"
TOP_K = 4
import time
OUTDIR = os.path.join(OUTDIR, time.strftime("run_%Y%m%d_%H%M%S"))

# Çalışma modu
ID_COL = "Istasyon_No"
RUN_ALL_STATIONS = True     # hızlı kullanımda False
STATION_ID = None             # None ise tek istasyonda ilkini otomatik seçer
PLOTS_FOR_ALL = True          # tüm istasyonlarda figür (RUN_ALL_STATIONS=True ise ağır olabilir)
MAKE_PLOTS_SINGLE = True      # tek istasyon modunda figür üret

# Hız/kalite ayarları
RANDOM_STATE = 42
FAST_MODE = False              # en hızlı mod
CV_SPLITS = 3 if FAST_MODE else 5
N_ITER = 6 if FAST_MODE else 14
CV_GAP = 36                   # pencere ≈36 ay → CV'de güvenlik boşluğu

# Variant aileleri (WF açık, WD/WPD kapalı; istersen True yap)
ENABLE_WF  = True
ENABLE_WD  = True
ENABLE_WPD = True

# Modeller: yalnızca HGBR + SVR açık (diğerlerini True yapabilirsin)
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

# Wavelet listeleri (FAST'te dar)
WAVELETS_DB  = ["db3","db4"] if FAST_MODE else ["db2","db3","db4","db5","db6","db8","db10"]
WAVELETS_SYM = ["sym4"] if FAST_MODE else ["sym4","sym5"]
WAVELETS_COI = [] if FAST_MODE else ["coif3","coif4"]  # FAST'te kapalı

# Gecikme yapıları (M01..M08) — hızlıda 3 yapı yeter
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

# Joblib temp klasörü (Windows memmap uyarılarını azaltır)
JOBLIB_TEMP = os.path.join(OUTDIR, "joblib_tmp")
os.makedirs(JOBLIB_TEMP, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", JOBLIB_TEMP)

# ---------------- Yayın stili ----------------
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "lines.linewidth": 1.4,
})

def hide_spines(ax):
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass

# -------------- Yardımcılar --------------
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
            raise ValueError(f"'{id_col}' sütunu bulunamadı ve otomatik eşleşme yapılamadı.")
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
        ts=ts.set_index("date").reindex(full); ts.index.name="date"; ts=ts.reset_index().rename(columns={"index":"date"})

        label = f"{sid}" if col_name is None else f"{sid} | {str(sub[col_name].iloc[0])}"
        stations[str(sid)]={"label":label, "df":ts}
    return stations, id_col

# -------------- SPI --------------
def compute_spi(df, value_col="precip", date_col="date", scale=12):
    s=df[[date_col,value_col]].copy()
    s[value_col]=pd.to_numeric(s[value_col],errors="coerce").clip(lower=0.0)
    s=s.dropna(subset=[value_col]).sort_values(date_col)
    s["month"]=pd.to_datetime(s[date_col]).dt.month
    s["agg"]=s[value_col].rolling(window=scale, min_periods=scale).sum()
    spi=pd.Series(index=s.index, dtype=float)
    for m in range(1,13):
        idx=(s["month"]==m) & s["agg"].notna()
        vals=s.loc[idx,"agg"].values
        if len(vals)<12: continue
        H=(vals==0).sum()/len(vals)
        pos=vals[vals>0.0]
        if len(pos)<6: continue
        try:
            a,loc,b=sp_gamma.fit(pos, floc=0.0)
        except Exception: continue
        G=sp_gamma.cdf(vals,a,loc=0.0,scale=b) if a>0 and b>0 else np.zeros_like(vals)
        p=np.clip(H+(1.0-H)*G, 1e-6, 1-1e-6)
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

def build_variant_frames(df_spi, target_col):
    base=df_spi.rename(columns={target_col:"y"}).copy()
    base=add_lags_max(base,"y","lag",MAX_LAG)
    frames={}

    # ileri yönlü doldurma (leakage riski düşük)
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

    # Ortak geçerli tarih kümesi
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
    return frames

# -------------- Metrikler --------------
def rmse(y_true,y_pred): return float(np.sqrt(mean_squared_error(y_true,y_pred)))
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
                KGE=kge_2009(y_true,y_pred))
from sklearn.metrics import make_scorer
RMSE_SCORER = make_scorer(lambda yt, yp: -rmse(yt, yp))

# -------------- Modeller ve arama uzayı --------------
def model_space(name):
    if name=="SVR":
        est = Pipeline([("scaler",StandardScaler()),("svm",SVR(kernel="rbf"))])
        space={"svm__C":np.logspace(0,3,10),
               "svm__gamma":np.logspace(-4,-1,8),
               "svm__epsilon":np.linspace(0.01,0.2,5)}
    elif name=="RF":
        est = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)   # nested paralellik yok
        space={"n_estimators":np.linspace(200,400,4,dtype=int),
               "max_depth":[None,6,10],
               "min_samples_split":[2,5],
               "min_samples_leaf":[1,2,4],
               "max_features":["sqrt",0.7]}
    elif name=="ET":
        est = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1)
        space={"n_estimators":np.linspace(200,400,4,dtype=int),
               "max_depth":[None,6,10],
               "min_samples_split":[2,5],
               "min_samples_leaf":[1,2,4],
               "max_features":["sqrt",0.7]}
    elif name=="GBR":
        est = GradientBoostingRegressor(random_state=RANDOM_STATE)
        space={"n_estimators":np.linspace(150,300,4,dtype=int),
               "learning_rate":np.logspace(-3,-0.6,6),
               "max_depth":[2,3,4],
               "min_samples_leaf":[1,2,4],
               "subsample":[0.7,1.0]}
    elif name=="HGBR":
        est = HistGradientBoostingRegressor(random_state=RANDOM_STATE, early_stopping=False)
        space={"learning_rate":np.logspace(-3,-0.3,8),
               "max_depth":[None,3,5],
               "max_leaf_nodes":[31,63,127],
               "min_samples_leaf":[1,5,10],
               "l2_regularization":np.logspace(-4,1,6)}
    elif name=="RIDGE":
        est = Pipeline([("scaler",StandardScaler()),("ridge",Ridge())])
        space={"ridge__alpha":np.logspace(-3,3,12),
               "ridge__fit_intercept":[True,False]}
    elif name=="KNN":
        est = Pipeline([("scaler",StandardScaler()),("knn",KNeighborsRegressor())])
        space={"knn__n_neighbors":np.arange(2,21),
               "knn__weights":["uniform","distance"],
               "knn__p":[1,2]}
    elif name=="LSVR":
        est = Pipeline([("scaler",StandardScaler()),("lsvr",LinearSVR(random_state=RANDOM_STATE,max_iter=10000))])
        space={"lsvr__C":np.logspace(-3,2,10),
               "lsvr__epsilon":np.linspace(0.0,0.2,5),
               "lsvr__loss":["epsilon_insensitive","squared_epsilon_insensitive"]}
    else:
        raise ValueError(name)
    return est,space

CANDIDATES = [m for m,on in MODELS_ON.items() if on]

def fit_cv(est, space, X, y, gap=CV_GAP):
    try:
        cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=gap)
    except TypeError:
        cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    rs = RandomizedSearchCV(est, space, n_iter=N_ITER, random_state=RANDOM_STATE,
                            cv=cv, scoring=RMSE_SCORER, n_jobs=-1, verbose=0)
    # süreç tabanlı backend + iç thread=1 + sabit temp
    with parallel_backend("loky", inner_max_num_threads=1, temp_folder=JOBLIB_TEMP):
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

def taylor_2d(obs, pred_map, title, savepath):
    so=np.std(obs,ddof=0)
    fig=plt.figure(figsize=(8.6,8.4)); ax=plt.subplot(111, polar=True)
    ax.set_theta_direction(-1); ax.set_theta_zero_location("E")
    ax.set_title(title, y=1.06)

    stats={}; max_std=so
    for lbl,pr in pred_map.items():
        r=np.corrcoef(obs,pr)[0,1]; sm=np.std(pr,ddof=0)
        if not np.isfinite([r,sm]).all(): continue
        stats[lbl]=(sm,r); max_std=max(max_std,sm)
    max_std*=1.3; ax.set_rlim(0,max_std)

    for c in [0.2,0.4,0.6,0.8,0.9,0.95,0.99,1.0]:
        th=np.arccos(np.clip(c,-1,1)); ax.plot([th,th],[0,max_std], color="#d9d9d9", lw=0.8)
        ax.text(th, max_std*1.01, f"{c:.2f}", ha="center", va="bottom", fontsize=8)

    th=np.linspace(0,np.pi/2,360)
    ax.plot(th, np.full_like(th, so), "--", color="gray", lw=1.2, label=f"Std(Obs)={so:.2f}")

    rmsd_levels=[0.25*so, 0.5*so, 0.75*so, 1.0*so]
    for e in rmsd_levels:
        phi=np.linspace(0,2*np.pi,800)
        x=so + e*np.cos(phi); y=e*np.sin(phi)
        r=np.sqrt(x**2+y**2); theta=np.arctan2(y,x)
        mask=(theta>=0)&(theta<=np.pi/2)
        ax.plot(theta[mask], r[mask], color="#bbbbbb", lw=0.8)
        ax.text(np.deg2rad(10), so+e+0.02*max_std, f"RMSD={e:.2f}", color="#777", fontsize=8)

    colors=plt.cm.Set2(np.linspace(0,1,len(stats)))
    for i,(lbl,(sm,r)) in enumerate(stats.items()):
        theta=np.arccos(np.clip(r,-1,1))
        ax.plot([theta],[sm], marker="o", ms=9, color=colors[i], label=f"{lbl} (r={r:.2f}, σ={sm:.2f})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.40,1.10), frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(savepath, dpi=300, bbox_inches="tight"); plt.close(fig)

def taylor_3d(obs, pred_map, title, savepath):
    so=np.std(obs,ddof=0)
    fig=plt.figure(figsize=(9.2,7.6)); ax=fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    rmsd_levels=[0.25*so,0.5*so,0.75*so,1.0*so]
    for e in rmsd_levels:
        phi=np.linspace(0,2*np.pi,600)
        x=so+e*np.cos(phi); y=0+e*np.sin(phi); z=np.full_like(x,e)
        ax.plot(x,y,z,color="#bbbbbb",lw=0.8,alpha=0.9)
        ax.text(so+e*0.95,0.0,e,f"RMSD={e:.2f}",color="#666",fontsize=8)
    th=np.linspace(0,2*np.pi,400)
    ax.plot(so*np.cos(th), so*np.sin(th), np.zeros_like(th), "--", color="gray", lw=1.2, alpha=0.9, label=f"Std(Obs)={so:.2f}")
    colors=plt.cm.Set2(np.linspace(0,1,len(pred_map)))
    for i,(lbl,pr) in enumerate(pred_map.items()):
        r=np.corrcoef(obs,pr)[0,1]; sm=np.std(pr,ddof=0)
        e=np.sqrt(so**2 + sm**2 - 2*so*sm*r)
        x=sm*r; y=sm*np.sqrt(max(0.0,1-r**2))
        ax.scatter(x,y,e,s=60,color=colors[i],depthshade=True,label=f"{lbl} (r={r:.2f}, σ={sm:.2f}, E={e:.2f})")
        ax.text(x,y,e,f" {lbl}",fontsize=9,color=colors[i])
    ax.set_xlabel("σ_m·r"); ax.set_ylabel("σ_m·√(1−r²)"); ax.set_zlabel("RMSD")
    ax.view_init(elev=24, azim=-45); ax.legend(loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(savepath, dpi=300, bbox_inches="tight"); plt.close(fig)

def rolling_rmse(y, yhat, win=24):
    e=(yhat-y)**2
    return pd.Series(e).rolling(win).mean().apply(np.sqrt).values

# Baseline yardımcı
def _naive_vec(y, m=1):
    y=np.asarray(y,dtype=float)
    yhat=np.full_like(y,np.nan,dtype=float)
    if m<len(y): yhat[m:]=y[:-m]
    return yhat

# -------------- Değerlendirme çekirdeği --------------
def evaluate_station(label, df_date_precip, make_plots=True):
    os.makedirs(OUTDIR, exist_ok=True)

    spi=compute_spi(df_date_precip, value_col="precip", date_col="date", scale=SPI_SCALE)
    data=df_date_precip[["date"]].merge(spi, on="date", how="left").rename(columns={f"spi_{SPI_SCALE}":"y"})
    variants=build_variant_frames(data[["date","y"]].copy(), target_col="y")

    any_key=list(variants.keys())[0]
    vdf0=variants[any_key][["date","y"]].dropna().reset_index(drop=True)
    y0=vdf0["y"].values; d0=vdf0["date"].values
    split0=int(0.7*len(vdf0))

    results=[]; preds_map={}; feature_names_map={}; best_est_map={}; best_params_list=[]
    # Naive-1/12 baseline
    if split0>=1 and len(vdf0)-split0>=1:
        for nm,m in [("Naive-1",1),("Naive-12",12)]:
            yhat=_naive_vec(y0,m=m)[split0:]; yte=y0[split0:]
            mask=np.isfinite(yhat)&np.isfinite(yte)
            if mask.sum()>=3:
                mets=metrics_all(yte[mask], yhat[mask])
                base_label=f"{nm} | NW | BASE"
                results.append({"Label":base_label,"Model":nm,"Variant":"NW","Struct":"BASE","CV_RMSE":np.nan,**mets})
                preds_map[base_label]=(d0[split0:], yte, yhat)
                feature_names_map[base_label]=[]; best_est_map[base_label]=None
                print(f"{label} → {base_label:>22s}  RMSE={mets['RMSE']:.3f}  KGE={mets['KGE']:.3f}  R={mets['R']:.3f}")

    for vname, vdf in variants.items():
        base_cols=[c for c in vdf.columns if c not in ["date","y"] and not c.startswith("lag") and not c.startswith("dnlag")]
        for mcode, m_lags in LAG_STRUCTURES.items():
            lag_cols = [f"dnlag{L}" for L in m_lags] if vname.startswith("WD") else [f"lag{L}" for L in m_lags]
            feat_cols=lag_cols + base_cols
            df_use=vdf[["date","y"]+feat_cols].dropna()
            if len(df_use)<120: continue
            X=df_use[feat_cols].values; y=df_use["y"].values; dates=df_use["date"].values
            split_local=int(0.7*len(df_use))
            Xtr,Xte=X[:split_local],X[split_local:]; ytr,yte=y[:split_local],y[split_local:]
            for mname in CANDIDATES:
                est,space=model_space(mname)
                best,cv_rmse,params=fit_cv(est,space,Xtr,ytr,gap=CV_GAP)
                yhat=best.predict(Xte)
                mets=metrics_all(yte,yhat)
                full_label=f"{mname} | {vname} | {mcode}"
                results.append({"Label":full_label,"Model":mname,"Variant":vname,"Struct":mcode,"CV_RMSE":cv_rmse,**mets})
                preds_map[full_label]=(dates[split_local:], yte, yhat)
                feature_names_map[full_label]=feat_cols
                best_est_map[full_label]=best
                best_params_list.append({"Label":full_label, **params, "Features":";".join(feat_cols)})
                print(f"{label} → {full_label:>22s}  RMSE={mets['RMSE']:.3f}  KGE={mets['KGE']:.3f}  R={mets['R']:.3f}")

    res=pd.DataFrame(results)
    if res.empty: raise RuntimeError(f"{label}: geçerli kombinasyon üretilemedi (veri azlığı?).")

    key="KGE" if SELECT_BY.upper()=="KGE" else "RMSE"
    asc=False if key=="KGE" else True
    topk=res.sort_values(key, ascending=asc).head(TOP_K)
    top_labels=topk["Label"].tolist()

    best_nw  = res[res["Variant"]=="NW"].sort_values(key, ascending=asc).iloc[0]
    non_nw   = res[res["Variant"]!="NW"].sort_values(key, ascending=asc)
    best_wave= non_nw.iloc[0] if not non_nw.empty else best_nw

    tag=_norm_text(label)[:40]
    os.makedirs(OUTDIR, exist_ok=True)
    res.to_csv(os.path.join(OUTDIR,f"{tag}_metrics_all_SPI{SPI_SCALE}_TOP4.csv"), index=False)
    topk.set_index("Label").loc[top_labels, ["Model","Variant","Struct","RMSE","KGE","R","R2","MAE"]].round(3)\
        .to_csv(os.path.join(OUTDIR,f"{tag}_top4_summary.csv"))
    if best_params_list:
        pd.DataFrame(best_params_list).to_csv(os.path.join(OUTDIR,f"{tag}_best_params.csv"), index=False)

    if make_plots:
        # Overlay
        fig,ax=plt.subplots(figsize=(12,4))
        d_ref,y_ref,_=preds_map[top_labels[0]]
        ax.plot(d_ref, y_ref, label="Gözlem", lw=1.3, color="#333")
        colors=plt.cm.Set2(np.linspace(0,1,len(top_labels)))
        for i,lbl in enumerate(top_labels):
            dte,yte,yhat=preds_map[lbl]
            ax.plot(dte, yhat, label=lbl, lw=1.4, color=colors[i])
        year_axis(ax, step=5); ax.axhline(0,color="#777",lw=0.8)
        ax.set_title(f"{label} | SPI-{SPI_SCALE} — En İyi {TOP_K}")
        ax.set_xlabel("Tarih"); ax.set_ylabel(f"SPI-{SPI_SCALE}"); ax.legend(frameon=False, ncol=2)
        hide_spines(ax)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"{tag}_TOP4_overlay.png"), dpi=300); plt.close(fig)

        # Taylor 2D/3D
        ref_y=preds_map[top_labels[0]][1]
        taylor_2d(ref_y, {lbl:preds_map[lbl][2] for lbl in top_labels},
                  f"{label} — Taylor (En İyi {TOP_K})",
                  os.path.join(OUTDIR,f"{tag}_TOP4_taylor.png"))
        taylor_3d(ref_y, {lbl:preds_map[lbl][2] for lbl in top_labels},
                  f"{label} — 3B Taylor (z=RMSD, En İyi {TOP_K})",
                  os.path.join(OUTDIR,f"{tag}_TOP4_taylor3D.png"))

        # Scatter 2x2
        fig,axes=plt.subplots(2,2,figsize=(12,10))
        for i,lbl in enumerate(top_labels):
            ax=axes[i//2,i%2]
            yte=preds_map[lbl][1]; yhat=preds_map[lbl][2]
            ax.scatter(yte,yhat,s=18,alpha=0.85,color=colors[i],label=lbl)
            lims=[min(np.nanmin(yte),np.nanmin(yhat)), max(np.nanmax(yte),np.nanmax(yhat))]
            ax.plot(lims,lims,"k--",lw=1.0,label="1:1")
            try:
                k,b=np.polyfit(yte,yhat,1); ax.plot(lims,[k*lims[0]+b,k*lims[1]+b], color="#444", lw=1.0, label=f"Reg: y={k:.2f}x+{b:.2f}")
            except Exception: pass
            row=topk[topk["Label"]==lbl].iloc[0]
            ax.set_title(f"{lbl}\nR={row['R']:.2f}, RMSE={row['RMSE']:.2f}, KGE={row['KGE']:.2f}")
            ax.set_xlabel("Gözlem"); ax.set_ylabel("Tahmin"); ax.legend(frameon=False, fontsize=8)
            hide_spines(ax)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"{tag}_TOP4_scatter.png"), dpi=300); plt.close(fig)

        # Artık serisi 2x2
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=False)
        for i,lbl in enumerate(top_labels):
            ax=axes[i//2,i%2]
            dte,yte,yhat=preds_map[lbl]
            ax.plot(dte, yhat-yte, lw=1.1, color=colors[i])
            ax.axhline(0,color="#333",lw=1.0); year_axis(ax, step=5)
            ax.set_title(f"Artık — {lbl}"); ax.set_xlabel("Tarih"); ax.set_ylabel("Artık")
            hide_spines(ax)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"{tag}_TOP4_residual_series.png"), dpi=300); plt.close(fig)

        # Artık dağılımı + KDE
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
        for i,lbl in enumerate(top_labels):
            ax=axes[i//2,i%2]; r=preds_map[lbl][2]-preds_map[lbl][1]; r=r[np.isfinite(r)]
            if len(r)==0: continue
            ax.hist(r,bins=30,density=True,alpha=0.45,color=colors[i],label="Hist")
            try:
                kde=sp_kde(r); xs=np.linspace(np.nanmin(r),np.nanmax(r),200)
                ax.plot(xs,kde(xs),lw=1.5,color=colors[i],label="KDE")
            except Exception: pass
            ax.axvline(0,color="#333",lw=1.0)
            ax.set_title(f"Artık Dağılımı — {lbl}"); ax.set_xlabel("Artık"); ax.set_ylabel("Yoğunluk")
            hide_spines(ax)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"{tag}_TOP4_residual_hist.png"), dpi=300); plt.close(fig)

        # QQ-plot 2x2
        fig,axes=plt.subplots(2,2,figsize=(12,10))
        for i,lbl in enumerate(top_labels):
            ax=axes[i//2,i%2]; r=preds_map[lbl][2]-preds_map[lbl][1]; r=r[np.isfinite(r)]
            if len(r)<3: continue
            (osm, osr), (slope, intercept, r_) = sp_probplot(r, dist="norm")
            ax.scatter(osm, osr, s=14, color=colors[i], alpha=0.85)
            ax.plot(osm, slope*osm + intercept, "k--", lw=1.0)
            ax.set_title(f"QQ-Plot — {lbl}"); ax.set_xlabel("Teorik"); ax.set_ylabel("Gözlenen")
            hide_spines(ax)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"{tag}_TOP4_qqplot.png"), dpi=300); plt.close(fig)

        # Kayan metrikler
        win=24
        fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
        for i,lbl in enumerate(top_labels):
            dte,yte,yhat=preds_map[lbl]
            axes[0].plot(dte, rolling_rmse(yte,yhat,win), lw=1.2, label=lbl, color=colors[i])
        axes[0].set_ylabel(f"RMSE (win={win})"); axes[0].set_title("Kayan RMSE"); axes[0].legend(frameon=False, ncol=2); hide_spines(axes[0])
        for i,lbl in enumerate(top_labels):
            dte,yte,yhat=preds_map[lbl]
            try:
                axes[1].plot(dte, pd.Series(yte).rolling(win).corr(pd.Series(yhat)).values, lw=1.2, label=lbl, color=colors[i])
            except Exception: pass
        axes[1].set_ylabel(f"Corr (win={win})"); axes[1].set_xlabel("Tarih"); axes[1].set_title("Kayan Korelasyon")
        year_axis(axes[1], step=5); hide_spines(axes[1])
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"{tag}_TOP4_rolling_metrics.png"), dpi=300); plt.close(fig)

        # Özet metrik barları
        mlist=top_labels
        rows=topk.set_index("Label").loc[mlist, ["RMSE","KGE","R2","MAE"]]
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        bcols=[colors[i] for i in range(len(mlist))]
        axes[0,0].bar(mlist, rows["RMSE"], color=bcols); axes[0,0].set_title("RMSE (↓)"); set_xtick_rotation(axes[0,0], 20, 'right'); hide_spines(axes[0,0])
        axes[0,1].bar(mlist, rows["KGE"], color=bcols); axes[0,1].set_title("KGE (↑)"); axes[0,1].set_ylim(-1,1); set_xtick_rotation(axes[0,1], 20, 'right'); hide_spines(axes[0,1])
        axes[1,0].bar(mlist, rows["R2"],  color=bcols); axes[1,0].set_title("R² (↑)"); axes[1,0].set_ylim(0,1); set_xtick_rotation(axes[1,0], 20, 'right'); hide_spines(axes[1,0])
        axes[1,1].bar(mlist, rows["MAE"], color=bcols); axes[1,1].set_title("MAE (↓)"); set_xtick_rotation(axes[1,1], 20, 'right'); hide_spines(axes[1,1])
        fig.suptitle(f"Özet Metrikler — En İyi {TOP_K} Kombinasyon", y=1.02)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"{tag}_TOP4_metric_bars.png"), dpi=300); plt.close(fig)

    return {
        "station": label,
        "best_overall": topk.iloc[0].to_dict(),
        "best_nw": best_nw.to_dict(),
        "best_wave": best_wave.to_dict(),
        "topk": topk.copy()
    }

# -------------- Ana akış --------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Girdi bulunamadı: {INPUT_XLSX}")

    stations_map, real_id_col = load_by_station_id(INPUT_XLSX, ID_COL)
    id_list=list(stations_map.keys())
    print(f"Toplam istasyon: {len(id_list)} | ID sütunu: {real_id_col}")

    if RUN_ALL_STATIONS:
        summaries=[]
        for sid in id_list:
            st=stations_map[sid]
            summ=evaluate_station(st["label"], st["df"], make_plots=PLOTS_FOR_ALL)
            summaries.append(summ)
        # Toplu özet CSV ve bar grafikleri
        rows=[]
        for s in summaries:
            def pick(d, fields=("RMSE","KGE","R2","MAE","Variant","Struct","Model","Label")):
                return {k: d.get(k, np.nan) for k in fields}
            best_all=pick(s["best_overall"]); best_nw=pick(s["best_nw"]); best_wv=pick(s["best_wave"])
            rows.append({
                "Station": s["station"],
                "BestAll_Model": best_all["Model"], "BestAll_Variant": best_all["Variant"], "BestAll_Struct": best_all["Struct"],
                "BestAll_RMSE": best_all["RMSE"], "BestAll_KGE": best_all["KGE"], "BestAll_R2": best_all["R2"], "BestAll_MAE": best_all["MAE"],
                "BestNW_Model": best_nw["Model"], "BestNW_Variant": best_nw["Variant"], "BestNW_Struct": best_nw["Struct"],
                "BestNW_RMSE": best_nw["RMSE"], "BestNW_KGE": best_nw["KGE"],
                "BestWV_Model": best_wv["Model"], "BestWV_Variant": best_wv["Variant"], "BestWV_Struct": best_wv["Struct"],
                "BestWV_RMSE": best_wv["RMSE"], "BestWV_KGE": best_wv["KGE"],
                "Delta_KGE": (best_wv["KGE"] - best_nw["KGE"]) if np.isfinite([best_wv["KGE"],best_nw["KGE"]]).all() else np.nan,
                "Delta_RMSE": (best_wv["RMSE"] - best_nw["RMSE"]) if np.isfinite([best_wv["RMSE"],best_nw["RMSE"]]).all() else np.nan,
            })
        comp=pd.DataFrame(rows); comp.to_csv(os.path.join(OUTDIR,"ALL_stations_summary.csv"), index=False)
        print("\nALL_stations_summary.csv yazıldı.")
        try:
            # KGE bar
            tmp=comp.sort_values("BestAll_KGE", ascending=False)
            fig,ax=plt.subplots(figsize=(max(9, len(tmp)*0.5), 5))
            ax.bar(tmp["Station"], tmp["BestAll_KGE"]); ax.set_title("En İyi Genel KGE — Tüm İstasyonlar"); ax.set_ylabel("KGE (↑)")
            set_xtick_rotation(ax,45,'right'); hide_spines(ax)
            fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"ALL_best_KGE_bar.png"), dpi=300); plt.close(fig)
            # RMSE bar
            tmp=comp.sort_values("BestAll_RMSE", ascending=True)
            fig,ax=plt.subplots(figsize=(max(9, len(tmp)*0.5), 5))
            ax.bar(tmp["Station"], tmp["BestAll_RMSE"]); ax.set_title("En İyi Genel RMSE — Tüm İstasyonlar"); ax.set_ylabel("RMSE (↓)")
            set_xtick_rotation(ax,45,'right'); hide_spines(ax)
            fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"ALL_best_RMSE_bar.png"), dpi=300); plt.close(fig)
            # Delta KGE
            fig,ax=plt.subplots(figsize=(max(9, len(comp)*0.5), 5))
            ax.bar(comp["Station"], comp["Delta_KGE"]); ax.axhline(0,color="#333",lw=1.0)
            ax.set_title("Wavelet Etkisi — ΔKGE (Wavelet − NW)"); ax.set_ylabel("ΔKGE"); set_xtick_rotation(ax,45,'right'); hide_spines(ax)
            fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"ALL_delta_KGE_bar.png"), dpi=300); plt.close(fig)
            # Delta RMSE
            fig,ax=plt.subplots(figsize=(max(9, len(comp)*0.5), 5))
            ax.bar(comp["Station"], comp["Delta_RMSE"]); ax.axhline(0,color="#333",lw=1.0)
            ax.set_title("Wavelet Etkisi — ΔRMSE (Wavelet − NW)"); ax.set_ylabel("ΔRMSE (↓ iyi ≈ negatif)"); set_xtick_rotation(ax,45,'right'); hide_spines(ax)
            fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"ALL_delta_RMSE_bar.png"), dpi=300); plt.close(fig)
        except Exception as e:
            print("[INFO] Toplu figürlerde hata/atlandı:", e)
    else:
        # Tek istasyon
        if STATION_ID is None:
            sid=id_list[0]
            print(f"[INFO] STATION_ID belirtilmedi; ilk istasyon kullanılıyor: {sid}")
        else:
            sid=str(STATION_ID)
            if sid not in stations_map:
                print(f"[WARN] STATION_ID={STATION_ID} bulunamadı, ilk istasyon kullanılacak.")
                sid=id_list[0]
        st=stations_map[sid]
        _=evaluate_station(st["label"], st["df"], make_plots=MAKE_PLOTS_SINGLE)
        print("\nTek istasyon çalışması tamamlandı.")

if __name__=="__main__":
    main()
