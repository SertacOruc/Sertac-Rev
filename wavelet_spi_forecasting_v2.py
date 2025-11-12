#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wavelet-Enhanced SPI Forecasting System (Q1 Research Grade)
============================================================
Advanced drought forecasting using wavelet analysis and machine learning.

Features:
  - Multi-scale SPI computation with robust calibration
  - Comprehensive wavelet feature engineering (DWT, WPD)
  - Ensemble ML models with automated hyperparameter tuning
  - Publication-ready visualizations (Taylor diagrams, skill scores)
  - CLI interface with YAML configuration support
  - Parallel processing with intelligent caching
  - Complete reproducibility tracking

Author: Research Team
Version: 2.0.0 (Q1 Ready)
License: MIT
"""

# ==================== ENVIRONMENT SETUP ====================
import os
import sys

os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import warnings
warnings.filterwarnings("ignore")

# ==================== IMPORTS ====================
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import argparse
import logging
import time
import re
import unicodedata as ucd

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer
)
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.inspection import permutation_importance

try:
    from sklearn.inspection import PartialDependenceDisplay
    _HAS_PDP = True
except ImportError:
    _HAS_PDP = False

from scipy.stats import (
    gamma as sp_gamma,
    norm as sp_norm,
    gaussian_kde as sp_kde,
    probplot as sp_probplot,
    skew,
    kurtosis,
    shapiro
)
import pywt
from joblib import Parallel, delayed, parallel_backend

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    # Dummy tqdm
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from diskcache import Cache
    _HAS_CACHE = True
except ImportError:
    _HAS_CACHE = False

plt.ioff()

# ==================== CONSTANTS ====================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_SPLITS = 5
DEFAULT_CV_GAP = 36  # months
DEFAULT_MIN_TRAIN_SAMPLES = 120  # months (10 years)
DEFAULT_N_ITER_SEARCH = 14

# Wavelet analysis constants
WAVELET_WINDOW_SIZE = 36  # 3 years monthly window
WAVELET_DECOMPOSITION_LEVEL = 3
WAVELET_THRESHOLD_MODE = "soft"

# SPI computation constants
SPI_MIN_POSITIVE_SAMPLES = 6
SPI_GAMMA_LOCATION = 0.0
SPI_EPSILON = 1e-6
SPI_PROB_CLIP_MIN = 1e-6
SPI_PROB_CLIP_MAX = 1 - 1e-6

# Feature engineering constants
MAX_LAG_DEFAULT = 12
LAG_STRUCTURES_PRESET = {
    "M01": [1],
    "M02": [1, 2],
    "M03": [1, 2, 12],
    "M04": [1, 2, 11, 12],
    "M05": [1, 2, 3, 11, 12],
    "M06": [1, 2, 3, 6, 12],
    "M07": [1, 3, 6, 9, 12],
    "M08": [1, 2, 3, 4, 5, 6, 11, 12],
}

# Visualization constants
DEFAULT_DPI = 120
SAVE_DPI = 300
FIGURE_FORMATS = ["png"]
ROLLING_WINDOW_MONTHS = 24
TAYLOR_DIAGRAM_RMSD_LEVELS = [0.25, 0.5, 0.75, 1.0]

# Model names
MODEL_REGISTRY = [
    "HGBR", "SVR", "RF", "ET", "GBR", "RIDGE", "KNN", "LSVR"
]

# Wavelet families
WAVELETS_DB_FULL = ["db2", "db3", "db4", "db5", "db6", "db8", "db10"]
WAVELETS_SYM_FULL = ["sym4", "sym5"]
WAVELETS_COI_FULL = ["coif3", "coif4"]

WAVELETS_DB_FAST = ["db3", "db4"]
WAVELETS_SYM_FAST = ["sym4"]
WAVELETS_COI_FAST = []

# ==================== DATA CLASSES ====================
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Paths
    input_file: str
    output_dir: str
    id_column: str = "Istasyon_No"

    # SPI settings
    spi_scales: List[int] = None
    spi_calibration_mode: str = "all"  # 'all' | 'train_ratio' | 'fixed_range'
    spi_calib_train_ratio: float = 0.70
    spi_clim_start: Optional[str] = None
    spi_clim_end: Optional[str] = None

    # Forecasting
    forecast_leads: List[int] = None

    # Model selection
    select_by: str = "KGE"  # 'KGE' | 'RMSE'
    top_k: int = 4

    # Station processing
    run_all_stations: bool = True
    station_id: Optional[str] = None

    # Speed/quality trade-off
    fast_mode: bool = False
    random_state: int = DEFAULT_RANDOM_STATE
    cv_splits: int = DEFAULT_CV_SPLITS
    n_iter: int = DEFAULT_N_ITER_SEARCH
    cv_gap: int = DEFAULT_CV_GAP
    min_train_samples: int = DEFAULT_MIN_TRAIN_SAMPLES
    n_jobs: int = -1

    # Feature engineering
    enable_wf: bool = True
    enable_wd: bool = True
    enable_wpd: bool = True
    max_lag: int = MAX_LAG_DEFAULT
    lag_structures: Dict[str, List[int]] = None
    date_align_strategy: str = "intersection"  # 'intersection' | 'min_start'

    # Feature selection
    enable_feature_selection: bool = False
    feature_selection_mode: str = "variance"  # 'variance' | 'from_model'
    variance_threshold: float = 0.0

    # Models
    enabled_models: Dict[str, bool] = None

    # Wavelets
    wavelets_db: List[str] = None
    wavelets_sym: List[str] = None
    wavelets_coi: List[str] = None

    # Diagnostics & visualization
    make_plots: bool = True
    enable_month_skill_heatmap: bool = False
    enable_perm_importance: bool = False
    enable_pdp_ice: bool = False
    enable_quality_report: bool = False

    # Advanced
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    verbose: int = 1

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.spi_scales is None:
            self.spi_scales = [12]
        if self.forecast_leads is None:
            self.forecast_leads = [0]
        if self.lag_structures is None:
            self.lag_structures = (
                {k: v for k, v in list(LAG_STRUCTURES_PRESET.items())[:3]}
                if self.fast_mode
                else LAG_STRUCTURES_PRESET.copy()
            )
        if self.enabled_models is None:
            self.enabled_models = {m: True for m in MODEL_REGISTRY}
        if self.wavelets_db is None:
            self.wavelets_db = WAVELETS_DB_FAST if self.fast_mode else WAVELETS_DB_FULL
        if self.wavelets_sym is None:
            self.wavelets_sym = WAVELETS_SYM_FAST if self.fast_mode else WAVELETS_SYM_FULL
        if self.wavelets_coi is None:
            self.wavelets_coi = WAVELETS_COI_FAST if self.fast_mode else WAVELETS_COI_FULL
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.output_dir, ".cache")

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        if not _HAS_YAML:
            raise ImportError("PyYAML required for YAML config loading")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        if not _HAS_YAML:
            raise ImportError("PyYAML required for YAML config saving")
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ==================== LOGGING SETUP ====================
def setup_logging(output_dir: str, verbose: int = 1) -> logging.Logger:
    """
    Configure logging system.

    Args:
        output_dir: Directory for log files
        verbose: 0=WARNING, 1=INFO, 2=DEBUG

    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = level_map.get(verbose, logging.INFO)

    logger = logging.getLogger("wavelet_spi")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler
    log_file = os.path.join(output_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


# ==================== CACHING ====================
class CacheManager:
    """Disk-based cache for expensive computations."""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.enabled = enabled and _HAS_CACHE
        if self.enabled:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = Cache(cache_dir)
        else:
            self.cache = None

    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache."""
        if not self.enabled:
            return None
        return self.cache.get(key)

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        """Store in cache."""
        if self.enabled:
            self.cache.set(key, value, expire=expire)

    def clear(self):
        """Clear all cache."""
        if self.enabled:
            self.cache.clear()


# ==================== UTILITY FUNCTIONS ====================
def normalize_text(s: str) -> str:
    """
    Normalize Turkish text for safe filenames.

    Args:
        s: Input string

    Returns:
        Normalized ASCII-safe string
    """
    s = str(s).replace("\u0130", "I").replace("\u0131", "i")
    s = ucd.normalize("NFD", s)
    s = "".join(ch for ch in s if ucd.category(ch) != "Mn")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_")
    return s


def read_excel_safely(path: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
    """
    Read Excel with multiple engine fallbacks.

    Args:
        path: Path to Excel file
        sheet_name: Sheet to read

    Returns:
        DataFrame with data

    Raises:
        RuntimeError: If all engines fail
    """
    engines = ["openpyxl", "calamine"]
    if path.lower().endswith(".xls"):
        engines.append("xlrd")

    last_error = None
    for engine in engines:
        try:
            return pd.read_excel(path, sheet_name=sheet_name, engine=engine)
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Failed to read Excel file {path}: {last_error}")


def hide_spines(ax: plt.Axes):
    """Remove top and right spines from axis."""
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass


def year_axis(ax: plt.Axes, step: int = 5):
    """Configure x-axis for yearly ticks."""
    ax.xaxis.set_major_locator(mdates.YearLocator(base=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def set_xtick_rotation(ax: plt.Axes, rot: float = 45, ha: str = 'right'):
    """Rotate x-axis tick labels."""
    for label in ax.get_xticklabels():
        try:
            label.set_rotation(rot)
            label.set_horizontalalignment(ha)
        except Exception:
            pass


# ==================== MATPLOTLIB CONFIGURATION ====================
def configure_matplotlib():
    """Set publication-quality matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


# ==================== DATA LOADING ====================
def load_stations(
    path: str,
    id_col: str = "Istasyon_No",
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Load station data from Excel file.

    Args:
        path: Path to Excel file
        id_col: Column name for station ID
        logger: Logger instance

    Returns:
        Tuple of (stations_dict, actual_id_column_name)
        stations_dict format: {
            "station_id": {
                "label": "ID | Name",
                "df": DataFrame with columns [date, precip]
            }
        }
    """
    if logger:
        logger.info(f"Loading stations from: {path}")

    df = read_excel_safely(path)

    # Find ID column
    if id_col not in df.columns:
        cols_norm = {normalize_text(c).upper(): c for c in df.columns}
        candidates = ["ISTASYON_NO", "ISTASYONNO", "STATION_NO", "NO", "IST_NO"]
        id_col_actual = None
        for cand in candidates:
            if cand in cols_norm:
                id_col_actual = cols_norm[cand]
                break
        if id_col_actual is None:
            raise ValueError(f"ID column '{id_col}' not found and no alternative matched")
        id_col = id_col_actual

    # Find other columns
    cols_norm = {normalize_text(c).upper(): c for c in df.columns}

    def find_col(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand in cols_norm:
                return cols_norm[cand]
        return None

    col_name = find_col("ISTASYON_ADI", "STATION_NAME", "ISTASYON")
    col_year = find_col("YIL", "YEAR")
    col_month = find_col("AY", "MONTH", "AY_NO", "MONTH_NO")

    # Find precipitation column
    val_col = None
    for norm_name, real_name in cols_norm.items():
        keywords = ["YAG", "RAIN", "PRECIP"]
        qualifiers = ["AYLIK", "TOPLAM", "TOTAL", "MONTH", "MM"]
        if any(k in norm_name for k in keywords) and any(q in norm_name for q in qualifiers):
            val_col = real_name
            break

    # Try "MANUEL" prefix
    if val_col is None:
        for norm_name, real_name in cols_norm.items():
            if "MANUEL" in norm_name and any(k in norm_name for k in ["YAG", "RAIN", "PRECIP"]):
                val_col = real_name
                break

    if val_col is None:
        raise ValueError("Monthly precipitation column not found")
    if col_year is None or col_month is None:
        raise ValueError("Year/month columns required")

    # Parse stations
    stations = {}
    for station_id, group in df.groupby(id_col):
        group = group.copy()

        # Extract year and month
        year = pd.to_numeric(
            group[col_year].astype(str).str.extract(r"(\d{4})")[0],
            errors="coerce"
        )
        month = pd.to_numeric(
            group[col_month].astype(str).str.extract(r"(\d{1,2})")[0],
            errors="coerce"
        )

        group = group.assign(_year=year, _month=month).dropna(subset=["_year", "_month"])
        group = group[
            (group["_month"] >= 1) & (group["_month"] <= 12) &
            (group["_year"].between(1800, 2100))
        ].copy()

        group["_year"] = group["_year"].astype(int)
        group["_month"] = group["_month"].astype(int)
        group["date"] = pd.to_datetime(
            group["_year"].astype(str) + "-" + group["_month"].astype(str).str.zfill(2) + "-01",
            errors="coerce"
        )
        group = group.dropna(subset=["date"]).sort_values("date")

        # Build time series
        ts = group[["date", val_col]].rename(columns={val_col: "precip"}).copy()
        ts["precip"] = pd.to_numeric(ts["precip"], errors="coerce").clip(lower=0.0)

        # Fill gaps
        full_range = pd.date_range(ts["date"].min(), ts["date"].max(), freq="MS")
        ts = ts.set_index("date").reindex(full_range)
        ts.index.name = "date"
        ts = ts.reset_index()

        # Label
        label = (
            f"{station_id}"
            if col_name is None
            else f"{station_id} | {str(group[col_name].iloc[0])}"
        )

        stations[str(station_id)] = {"label": label, "df": ts}

    if logger:
        logger.info(f"Loaded {len(stations)} stations")

    return stations, id_col


# ==================== SPI COMPUTATION ====================
def compute_spi(
    df: pd.DataFrame,
    value_col: str = "precip",
    date_col: str = "date",
    scale: int = 12,
    calib_mode: str = "all",
    calib_train_ratio: float = 0.7,
    calib_start: Optional[str] = None,
    calib_end: Optional[str] = None,
    min_pos_samples: int = SPI_MIN_POSITIVE_SAMPLES,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Compute Standardized Precipitation Index (McKee et al., 1993).

    SPI uses gamma distribution to fit precipitation and transforms to
    standard normal distribution.

    Args:
        df: DataFrame with precipitation data
        value_col: Column name for precipitation values
        date_col: Column name for dates
        scale: Accumulation period in months
        calib_mode: Calibration strategy ('all', 'train_ratio', 'fixed_range')
        calib_train_ratio: Training ratio for 'train_ratio' mode
        calib_start: Start date for 'fixed_range' mode
        calib_end: End date for 'fixed_range' mode
        min_pos_samples: Minimum positive samples for gamma fit
        logger: Logger instance

    Returns:
        DataFrame with columns [date, spi_{scale}]

    References:
        McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of
        drought frequency and duration to time scales. In Proceedings of the 8th
        Conference on Applied Climatology (Vol. 17, No. 22, pp. 179-183).
    """
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s[value_col] = pd.to_numeric(s[value_col], errors="coerce").clip(lower=0.0)
    s = s.dropna(subset=[value_col]).sort_values(date_col)
    s["month"] = s[date_col].dt.month
    s["agg"] = s[value_col].rolling(window=scale, min_periods=scale).sum()

    # Determine calibration period
    if calib_mode == "fixed_range":
        cal_start = pd.to_datetime(calib_start) if calib_start else None
        cal_end = pd.to_datetime(calib_end) if calib_end else None
    elif calib_mode == "train_ratio":
        valid = s.loc[s["agg"].notna(), date_col]
        if len(valid) > 0:
            k = max(1, int(np.floor(len(valid) * calib_train_ratio)))
            cal_end = pd.to_datetime(valid.iloc[k - 1])
            cal_start = None
        else:
            cal_start = cal_end = None
    else:  # 'all'
        cal_start = cal_end = None

    spi = pd.Series(index=s.index, dtype=float)

    # Fit gamma distribution per month
    for month in range(1, 13):
        idx = (s["month"] == month) & s["agg"].notna()
        vals = s.loc[idx, "agg"].values

        if len(vals) < 12:
            continue

        # Select calibration subset
        if calib_mode == "fixed_range":
            idx_cal = idx & (
                (cal_start is None or s[date_col] >= cal_start) &
                (cal_end is None or s[date_col] <= cal_end)
            )
        elif calib_mode == "train_ratio":
            idx_cal = idx & ((cal_end is None) | (s[date_col] <= cal_end))
        else:
            idx_cal = idx

        pos_cal = s.loc[idx_cal, "agg"].values
        pos_cal = pos_cal[pos_cal > 0.0]

        if len(pos_cal) < min_pos_samples:
            pos_cal = vals[vals > 0.0]

        # Zero probability
        H = (vals == 0).sum() / len(vals)
        pos = vals[vals > 0.0]

        if len(pos) < min_pos_samples:
            continue

        # Fit gamma distribution
        try:
            a, loc, b = sp_gamma.fit(pos_cal, floc=SPI_GAMMA_LOCATION)
            if a <= 0 or b <= 0:
                if logger:
                    logger.warning(f"Invalid gamma parameters for month {month}: a={a:.3f}, b={b:.3f}")
                continue
        except Exception as e:
            if logger:
                logger.warning(f"Gamma fit failed for month {month}: {e}")
            continue

        # Compute SPI
        G = sp_gamma.cdf(vals, a, loc=0.0, scale=b) if a > 0 and b > 0 else np.zeros_like(vals)
        p = np.clip(H + (1.0 - H) * G, SPI_PROB_CLIP_MIN, SPI_PROB_CLIP_MAX)
        spi.loc[idx] = sp_norm.ppf(p)

    out = s[[date_col]].copy()
    out[f"spi_{scale}"] = spi.values
    return out


# ==================== FEATURE ENGINEERING ====================
def add_lags(
    df: pd.DataFrame,
    target_col: str = "y",
    prefix: str = "lag",
    max_lag: int = MAX_LAG_DEFAULT
) -> pd.DataFrame:
    """
    Add lagged features.

    Args:
        df: Input DataFrame
        target_col: Column to lag
        prefix: Prefix for lag column names
        max_lag: Maximum lag order

    Returns:
        DataFrame with lag columns added
    """
    for lag in range(1, max_lag + 1):
        df[f"{prefix}{lag}"] = df[target_col].shift(lag)
    return df


def dwt_window_features(
    series: pd.Series,
    wavelet: str = "db4",
    level: int = WAVELET_DECOMPOSITION_LEVEL,
    window: int = WAVELET_WINDOW_SIZE
) -> pd.DataFrame:
    """
    Extract DWT features using sliding window.

    Computes energy, mean, std, max, and relative energy for each
    decomposition level (approximation and details).

    Args:
        series: Input time series
        wavelet: Wavelet name (e.g., 'db4', 'sym5')
        level: Decomposition level
        window: Window size in samples

    Returns:
        DataFrame with wavelet features

    Notes:
        Features are only computed for positions >= window to avoid lookahead bias.
    """
    s = pd.Series(series).astype(float).values
    n = len(s)

    # Get coefficient names
    _ = pywt.wavedec(np.zeros(window), wavelet, level=level)
    names = [f"A{level}"] + [f"D{k}" for k in range(level, 0, -1)]
    stats = ["E", "M", "S", "X", "rE"]

    # Initialize feature dict
    feats = {f"WF_{wavelet}_{nm}_{st}": [np.nan] * n for nm in names for st in stats}

    # Sliding window computation
    for i in range(window, n):
        arr = s[i - window:i]
        if np.isnan(arr).any():
            continue

        coeffs = pywt.wavedec(arr, wavelet, level=level)
        energies = []

        for nm, c in zip(names, coeffs):
            c = np.asarray(c)
            e = float(np.sum(c ** 2))
            energies.append(e)

            feats[f"WF_{wavelet}_{nm}_E"][i] = e
            feats[f"WF_{wavelet}_{nm}_M"][i] = float(np.mean(np.abs(c)))
            feats[f"WF_{wavelet}_{nm}_S"][i] = float(np.std(c))
            feats[f"WF_{wavelet}_{nm}_X"][i] = float(np.max(np.abs(c)))

        # Relative energy
        total_energy = float(np.sum(energies)) + 1e-12
        for nm, e in zip(names, energies):
            feats[f"WF_{wavelet}_{nm}_rE"][i] = e / total_energy

    return pd.DataFrame(feats)


def dwt_causal_denoise(
    series: pd.Series,
    wavelet: str = "db4",
    level: int = WAVELET_DECOMPOSITION_LEVEL,
    window: int = WAVELET_WINDOW_SIZE,
    mode: str = WAVELET_THRESHOLD_MODE
) -> pd.Series:
    """
    Causal wavelet denoising using sliding window.

    Applies soft/hard thresholding to detail coefficients within each window.
    Threshold is estimated from finest detail level using MAD (Median Absolute Deviation).

    Args:
        series: Input time series
        wavelet: Wavelet name
        level: Decomposition level
        window: Window size
        mode: Thresholding mode ('soft' or 'hard')

    Returns:
        Denoised series

    References:
        Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by
        wavelet shrinkage. Biometrika, 81(3), 425-455.
    """
    s = pd.Series(series).astype(float).values
    n = len(s)
    y_dn = np.full(n, np.nan)

    for i in range(window, n):
        arr = s[i - window:i]
        if np.isnan(arr).any():
            continue

        coeffs = pywt.wavedec(arr, wavelet, level=level)

        # Estimate noise std from finest detail
        d1 = coeffs[-1]
        sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 + 1e-12
        threshold = sigma * np.sqrt(2 * np.log(len(arr)))

        # Threshold detail coefficients
        new_coeffs = [coeffs[0]] + [
            pywt.threshold(c, value=threshold, mode=mode) for c in coeffs[1:]
        ]

        # Reconstruct
        rec = pywt.waverec(new_coeffs, wavelet)
        y_dn[i] = rec[-1]

    return pd.Series(y_dn)


def wpd_window_features(
    series: pd.Series,
    wavelet: str = "db4",
    level: int = WAVELET_DECOMPOSITION_LEVEL,
    window: int = WAVELET_WINDOW_SIZE
) -> pd.DataFrame:
    """
    Extract Wavelet Packet Decomposition features.

    WPD provides more detailed frequency analysis by decomposing both
    approximation and detail coefficients.

    Args:
        series: Input time series
        wavelet: Wavelet name
        level: Decomposition level
        window: Window size

    Returns:
        DataFrame with WPD features (energy and relative energy for each node)
    """
    s = pd.Series(series).astype(float).values
    n = len(s)

    # Get node names
    dummy = np.zeros(window)
    wp = pywt.WaveletPacket(data=dummy, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, order='natural')]

    # Initialize features
    feats = {}
    for path in nodes:
        feats[f"WPD_{wavelet}_{path}_E"] = [np.nan] * n
        feats[f"WPD_{wavelet}_{path}_rE"] = [np.nan] * n

    # Sliding window
    for i in range(window, n):
        arr = s[i - window:i]
        if np.isnan(arr).any():
            continue

        wp = pywt.WaveletPacket(data=arr, wavelet=wavelet, mode='symmetric', maxlevel=level)
        lvl = wp.get_level(level, order='natural')

        energies = [float(np.sum(np.asarray(node.data) ** 2)) for node in lvl]
        total_energy = float(np.sum(energies)) + 1e-12

        for e, node in zip(energies, lvl):
            feats[f"WPD_{wavelet}_{node.path}_E"][i] = e
            feats[f"WPD_{wavelet}_{node.path}_rE"][i] = e / total_energy

    return pd.DataFrame(feats)


def build_variant_frames(
    df_spi: pd.DataFrame,
    target_col: str,
    config: ExperimentConfig,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Build multiple feature variant DataFrames.

    Variants:
      - NW: No wavelets (baseline with lags only)
      - WF-{wavelet}: Wavelet features (DWT statistics)
      - WD-{wavelet}: Wavelet-denoised lags
      - WPD-{wavelet}: Wavelet packet features

    Args:
        df_spi: DataFrame with SPI values
        target_col: Target column name
        config: Experiment configuration
        logger: Logger instance

    Returns:
        Dictionary mapping variant name to DataFrame
    """
    if logger:
        logger.debug(f"Building variant frames for {target_col}")

    base = df_spi.rename(columns={target_col: "y"}).copy()
    base = add_lags(base, "y", "lag", config.max_lag)

    frames = {}

    # Interpolate for wavelet computations (forward fill to avoid lookahead)
    y_interp = base["y"].interpolate(limit_direction="forward")

    # No-wavelet baseline
    frames["NW"] = base.copy()

    # Wavelet features
    if config.enable_wf:
        wf_wavelets = config.wavelets_db + config.wavelets_sym + config.wavelets_coi
        if logger:
            logger.debug(f"Computing WF features for {len(wf_wavelets)} wavelets")

        for wav in wf_wavelets:
            wf = dwt_window_features(
                y_interp.shift(1).fillna(y_interp.median()),
                wavelet=wav,
                level=WAVELET_DECOMPOSITION_LEVEL,
                window=WAVELET_WINDOW_SIZE
            )
            frames[f"WF-{wav}"] = pd.concat([base, wf], axis=1)

    # Wavelet denoising
    if config.enable_wd:
        wd_wavelets = config.wavelets_db + config.wavelets_sym + config.wavelets_coi
        if logger:
            logger.debug(f"Computing WD features for {len(wd_wavelets)} wavelets")

        for wav in wd_wavelets:
            ydn = dwt_causal_denoise(
                y_interp,
                wavelet=wav,
                level=WAVELET_DECOMPOSITION_LEVEL,
                window=WAVELET_WINDOW_SIZE,
                mode=WAVELET_THRESHOLD_MODE
            )
            dn = pd.DataFrame({"y_dn": ydn})
            dn = add_lags(dn, "y_dn", "dnlag", config.max_lag)
            frames[f"WD-{wav}"] = pd.concat(
                [base[["date", "y"]], dn.drop(columns=["y_dn"])],
                axis=1
            )

    # Wavelet packet decomposition
    if config.enable_wpd:
        wpd_wavelets = (
            (config.wavelets_db[:2] + ["sym4"])
            if config.fast_mode
            else (config.wavelets_db + config.wavelets_sym)
        )
        if logger:
            logger.debug(f"Computing WPD features for {len(wpd_wavelets)} wavelets")

        for wav in wpd_wavelets:
            wp = wpd_window_features(
                y_interp.shift(1).fillna(y_interp.median()),
                wavelet=wav,
                level=WAVELET_DECOMPOSITION_LEVEL,
                window=WAVELET_WINDOW_SIZE
            )
            frames[f"WPD-{wav}"] = pd.concat([base, wp], axis=1)

    # Date alignment
    if config.date_align_strategy == "intersection":
        # Use only dates valid across all variants
        date_sets = []
        for _, variant_df in frames.items():
            req_cols = [c for c in variant_df.columns if c not in ["date", "y"]]
            mask = variant_df[["y"] + req_cols].notna().all(axis=1)
            date_sets.append(set(pd.to_datetime(variant_df.loc[mask, "date"]).astype("datetime64[ns]")))

        common_dates = sorted(set.intersection(*date_sets)) if date_sets else []

        for key, variant_df in frames.items():
            req_cols = [c for c in variant_df.columns if c not in ["date", "y"]]
            variant_df["date"] = pd.to_datetime(variant_df["date"])
            variant_df = variant_df[variant_df["date"].isin(common_dates)].copy()
            variant_df = variant_df[["date", "y"] + req_cols].sort_values("date").reset_index(drop=True)
            frames[key] = variant_df
    else:  # 'min_start'
        # Align to latest start date
        first_dates = []
        for _, variant_df in frames.items():
            req_cols = [c for c in variant_df.columns if c not in ["date", "y"]]
            mask = variant_df[["y"] + req_cols].notna().all(axis=1)
            d = pd.to_datetime(variant_df.loc[mask, "date"])
            if len(d) > 0:
                first_dates.append(d.min())

        global_start = max(first_dates) if first_dates else None

        for key, variant_df in frames.items():
            req_cols = [c for c in variant_df.columns if c not in ["date", "y"]]
            variant_df["date"] = pd.to_datetime(variant_df["date"])
            if global_start is not None:
                variant_df = variant_df[variant_df["date"] >= global_start].copy()
            variant_df = variant_df[["date", "y"] + req_cols].dropna().sort_values("date").reset_index(drop=True)
            frames[key] = variant_df

    if logger:
        logger.debug(f"Built {len(frames)} variant frames")

    return frames


def apply_forecast_lead(
    frames: Dict[str, pd.DataFrame],
    lead: int
) -> Dict[str, pd.DataFrame]:
    """
    Apply forecast lead time to variants.

    Args:
        frames: Dictionary of variant DataFrames
        lead: Lead time in months (0 = concurrent)

    Returns:
        Modified frames with shifted target
    """
    if lead == 0:
        return frames

    out = {}
    for key, df in frames.items():
        d = df.copy()
        d["y_lead"] = d["y"].shift(-lead)
        d = d.drop(columns=["y"]).rename(columns={"y_lead": "y"})
        out[key] = d

    return out


# ==================== METRICS ====================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

    Range: (-∞, 1], 1 = perfect, 0 = mean baseline, <0 = worse than mean
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((y_pred - y_true) ** 2) / denom)


def willmott_d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Willmott's Index of Agreement.

    d = 1 - sum((yhat - y)^2) / sum((|yhat - mean(y)| + |y - mean(y)|)^2)

    Range: [0, 1], 1 = perfect agreement
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.sum((y_pred - y_true) ** 2)
    den = np.sum(
        (np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2
    )
    if den == 0:
        return np.nan
    return float(1.0 - num / den)


def kge_2009(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Kling-Gupta Efficiency (2009 version).

    KGE = 1 - sqrt((r-1)^2 + (β-1)^2 + (γ-1)^2)

    where:
      r = correlation coefficient
      β = bias ratio = mean(yhat) / mean(y)
      γ = variability ratio = (std(yhat)/mean(yhat)) / (std(y)/mean(y))

    Range: (-∞, 1], 1 = perfect, 0.41 ≈ mean baseline

    References:
        Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
        Decomposition of the mean squared error and NSE performance criteria:
        Implications for improving hydrological modelling. Journal of Hydrology,
        377(1-2), 80-91.
    """
    mu_o = np.mean(y_true)
    mu_g = np.mean(y_pred)
    s_o = np.std(y_true, ddof=0)
    s_g = np.std(y_pred, ddof=0)
    r = np.corrcoef(y_true, y_pred)[0, 1]

    if not np.isfinite([mu_o, mu_g, s_o, s_g, r]).all():
        return float("nan")

    if abs(mu_o) < eps or abs(mu_g) < eps or s_o == 0 or s_g == 0:
        return float(r)

    beta = mu_g / mu_o
    cv_o = s_o / mu_o
    cv_g = s_g / mu_g
    gamma = cv_g / cv_o if cv_o != 0 else np.nan

    return float(1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Observed values
        y_pred: Predicted values

    Returns:
        Dictionary with metric names and values
    """
    return {
        "R": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "KGE": kge_2009(y_true, y_pred),
        "NSE": nse(y_true, y_pred),
        "WD": willmott_d(y_true, y_pred),
    }


# Create scorer for sklearn
RMSE_SCORER = make_scorer(lambda yt, yp: -rmse(yt, yp))


# ==================== MODEL DEFINITIONS ====================
def create_feature_selector(config: ExperimentConfig):
    """Create feature selection transformer if enabled."""
    if not config.enable_feature_selection:
        return None

    if config.feature_selection_mode == "variance":
        return VarianceThreshold(threshold=config.variance_threshold)
    elif config.feature_selection_mode == "from_model":
        base = HistGradientBoostingRegressor(
            random_state=config.random_state,
            early_stopping=False
        )
        return SelectFromModel(
            estimator=base,
            threshold="median",
            max_features=None
        )
    else:
        return None


def get_model_pipeline(
    model_name: str,
    config: ExperimentConfig
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Get model pipeline and hyperparameter search space.

    Args:
        model_name: Model identifier (HGBR, SVR, RF, etc.)
        config: Experiment configuration

    Returns:
        Tuple of (pipeline, param_distributions)
    """
    use_scaler = model_name in {"SVR", "RIDGE", "KNN", "LSVR"}
    feat_selector = create_feature_selector(config)

    # Define base estimator and search space
    if model_name == "SVR":
        base = SVR(kernel="rbf")
        space = {
            "model__C": np.logspace(0, 3, 10),
            "model__gamma": np.logspace(-4, -1, 8),
            "model__epsilon": np.linspace(0.01, 0.2, 5),
        }
    elif model_name == "RF":
        base = RandomForestRegressor(random_state=config.random_state, n_jobs=1)
        space = {
            "model__n_estimators": np.linspace(200, 400, 4, dtype=int),
            "model__max_depth": [None, 6, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7],
        }
    elif model_name == "ET":
        base = ExtraTreesRegressor(random_state=config.random_state, n_jobs=1)
        space = {
            "model__n_estimators": np.linspace(200, 400, 4, dtype=int),
            "model__max_depth": [None, 6, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7],
        }
    elif model_name == "GBR":
        base = GradientBoostingRegressor(random_state=config.random_state)
        space = {
            "model__n_estimators": np.linspace(150, 300, 4, dtype=int),
            "model__learning_rate": np.logspace(-3, -0.6, 6),
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.7, 1.0],
        }
    elif model_name == "HGBR":
        base = HistGradientBoostingRegressor(
            random_state=config.random_state,
            early_stopping=False
        )
        space = {
            "model__learning_rate": np.logspace(-3, -0.3, 8),
            "model__max_depth": [None, 3, 5],
            "model__max_leaf_nodes": [31, 63, 127],
            "model__min_samples_leaf": [1, 5, 10],
            "model__l2_regularization": np.logspace(-4, 1, 6),
        }
    elif model_name == "RIDGE":
        base = Ridge()
        space = {
            "model__alpha": np.logspace(-3, 3, 12),
            "model__fit_intercept": [True, False],
        }
    elif model_name == "KNN":
        base = KNeighborsRegressor()
        space = {
            "model__n_neighbors": np.arange(2, 21),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    elif model_name == "LSVR":
        base = LinearSVR(random_state=config.random_state, max_iter=10000)
        space = {
            "model__C": np.logspace(-3, 2, 10),
            "model__epsilon": np.linspace(0.0, 0.2, 5),
            "model__loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Build pipeline
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if feat_selector is not None:
        steps.append(("feat_sel", feat_selector))
    steps.append(("model", base))

    pipeline = Pipeline(steps)
    return pipeline, space


def fit_with_cv(
    pipeline: Pipeline,
    param_space: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
    logger: Optional[logging.Logger] = None
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """
    Fit model with time series cross-validation.

    Args:
        pipeline: sklearn Pipeline
        param_space: Hyperparameter distributions
        X: Feature matrix
        y: Target vector
        config: Experiment configuration
        logger: Logger instance

    Returns:
        Tuple of (best_estimator, best_cv_rmse, best_params)
    """
    try:
        cv = TimeSeriesSplit(n_splits=config.cv_splits, gap=config.cv_gap)
    except TypeError:
        # Older sklearn versions don't have gap parameter
        cv = TimeSeriesSplit(n_splits=config.cv_splits)

    search = RandomizedSearchCV(
        pipeline,
        param_space,
        n_iter=config.n_iter,
        random_state=config.random_state,
        cv=cv,
        scoring=RMSE_SCORER,
        n_jobs=config.n_jobs,
        verbose=0
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        search.fit(X, y)

    best_rmse = -search.best_score_

    return search.best_estimator_, best_rmse, search.best_params_


# ==================== NAIVE BASELINES ====================
def naive_forecast(y: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Naive persistence forecast.

    Args:
        y: Time series
        lag: Persistence lag (1=last value, 12=seasonal)

    Returns:
        Forecast array
    """
    y = np.asarray(y, dtype=float)
    yhat = np.full_like(y, np.nan, dtype=float)
    if lag < len(y):
        yhat[lag:] = y[:-lag]
    return yhat


# ==================== CONTINUED IN NEXT MESSAGE ====================

# ==================== VISUALIZATION ====================
def taylor_diagram_2d(
    obs: np.ndarray,
    pred_map: Dict[str, np.ndarray],
    title: str,
    savepath: str
):
    """
    Create 2D Taylor diagram.

    Taylor diagrams display three statistics simultaneously:
    - Correlation (angle from x-axis)
    - Standard deviation (distance from origin)
    - RMSD (distance from reference point)

    Args:
        obs: Observed values
        pred_map: Dictionary mapping labels to predictions
        title: Plot title
        savepath: Output file path

    References:
        Taylor, K. E. (2001). Summarizing multiple aspects of model performance
        in a single diagram. Journal of Geophysical Research, 106(D7), 7183-7192.
    """
    s_obs = np.std(obs, ddof=0)

    fig = plt.figure(figsize=(8.6, 8.4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("E")
    ax.set_title(title, y=1.06)

    # Compute statistics
    stats = {}
    max_std = s_obs
    for label, pred in pred_map.items():
        r = np.corrcoef(obs, pred)[0, 1]
        s_pred = np.std(pred, ddof=0)
        if not np.isfinite([r, s_pred]).all():
            continue
        stats[label] = (s_pred, r)
        max_std = max(max_std, s_pred)

    max_std *= 1.3
    ax.set_rlim(0, max_std)

    # Correlation arcs
    for corr in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]:
        theta = np.arccos(np.clip(corr, -1, 1))
        ax.plot([theta, theta], [0, max_std], color="#d9d9d9", lw=0.8)
        ax.text(theta, max_std * 1.01, f"{corr:.2f}", ha="center", va="bottom", fontsize=8)

    # Reference std arc
    theta = np.linspace(0, np.pi / 2, 360)
    ax.plot(theta, np.full_like(theta, s_obs), "--", color="gray", lw=1.2, label=f"Std(Obs)={s_obs:.2f}")

    # RMSD contours
    rmsd_levels = [level * s_obs for level in TAYLOR_DIAGRAM_RMSD_LEVELS]
    for rmsd in rmsd_levels:
        phi = np.linspace(0, 2 * np.pi, 800)
        x = s_obs + rmsd * np.cos(phi)
        y = rmsd * np.sin(phi)
        r = np.sqrt(x ** 2 + y ** 2)
        theta_contour = np.arctan2(y, x)
        mask = (theta_contour >= 0) & (theta_contour <= np.pi / 2)
        ax.plot(theta_contour[mask], r[mask], color="#bbbbbb", lw=0.8)
        ax.text(np.deg2rad(10), s_obs + rmsd + 0.02 * max_std, f"RMSD={rmsd:.2f}", color="#777", fontsize=8)

    # Plot models
    colors = plt.cm.Set2(np.linspace(0, 1, len(stats)))
    for i, (label, (s_pred, r)) in enumerate(stats.items()):
        theta = np.arccos(np.clip(r, -1, 1))
        ax.plot([theta], [s_pred], marker="o", ms=9, color=colors[i], label=f"{label} (r={r:.2f}, σ={s_pred:.2f})")

    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.10), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def taylor_diagram_3d(
    obs: np.ndarray,
    pred_map: Dict[str, np.ndarray],
    title: str,
    savepath: str
):
    """
    Create 3D Taylor diagram with RMSD as z-axis.

    Args:
        obs: Observed values
        pred_map: Dictionary mapping labels to predictions
        title: Plot title
        savepath: Output file path
    """
    s_obs = np.std(obs, ddof=0)

    fig = plt.figure(figsize=(9.2, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # RMSD levels
    rmsd_levels = [level * s_obs for level in TAYLOR_DIAGRAM_RMSD_LEVELS]
    for rmsd in rmsd_levels:
        phi = np.linspace(0, 2 * np.pi, 600)
        x = s_obs + rmsd * np.cos(phi)
        y = rmsd * np.sin(phi)
        z = np.full_like(x, rmsd)
        ax.plot(x, y, z, color="#bbbbbb", lw=0.8, alpha=0.9)
        ax.text(s_obs + rmsd * 0.95, 0.0, rmsd, f"RMSD={rmsd:.2f}", color="#666", fontsize=8)

    # Reference circle
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(s_obs * np.cos(theta), s_obs * np.sin(theta), np.zeros_like(theta), "--", color="gray", lw=1.2, alpha=0.9, label=f"Std(Obs)={s_obs:.2f}")

    # Plot models
    colors = plt.cm.Set2(np.linspace(0, 1, len(pred_map)))
    for i, (label, pred) in enumerate(pred_map.items()):
        r = np.corrcoef(obs, pred)[0, 1]
        s_pred = np.std(pred, ddof=0)
        rmsd_val = np.sqrt(s_obs ** 2 + s_pred ** 2 - 2 * s_obs * s_pred * r)
        x = s_pred * r
        y = s_pred * np.sqrt(max(0.0, 1 - r ** 2))
        ax.scatter(x, y, rmsd_val, s=60, color=colors[i], depthshade=True, label=f"{label} (r={r:.2f}, σ={s_pred:.2f}, E={rmsd_val:.2f})")
        ax.text(x, y, rmsd_val, f" {label}", fontsize=9, color=colors[i])

    ax.set_xlabel("σ_m·r")
    ax.set_ylabel("σ_m·√(1−r²)")
    ax.set_zlabel("RMSD")
    ax.view_init(elev=24, azim=-45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def rolling_metric(
    y: np.ndarray,
    yhat: np.ndarray,
    window: int,
    metric: str = "rmse"
) -> np.ndarray:
    """
    Compute rolling window metric.

    Args:
        y: Observed values
        yhat: Predicted values
        window: Window size
        metric: Metric name ('rmse' or 'corr')

    Returns:
        Rolling metric values
    """
    if metric == "rmse":
        e = (yhat - y) ** 2
        return pd.Series(e).rolling(window).mean().apply(np.sqrt).values
    elif metric == "corr":
        s_y = pd.Series(y)
        s_yhat = pd.Series(yhat)
        return s_y.rolling(window).corr(s_yhat).values
    else:
        raise ValueError(f"Unknown metric: {metric}")


def align_predictions_by_common_dates(
    preds_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: List[str]
) -> Tuple[Optional[pd.DatetimeIndex], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Align predictions to common dates across models.

    Args:
        preds_map: Dictionary mapping labels to (dates, y_true, y_pred)
        labels: List of labels to align

    Returns:
        Tuple of (common_dates, obs_aligned, preds_aligned_dict)
    """
    date_sets = []
    per_label = {}

    for label in labels:
        dates, y_true, y_pred = preds_map[label]
        dates = pd.to_datetime(dates)
        per_label[label] = (dates, pd.Series(y_true, index=dates), pd.Series(y_pred, index=dates))
        date_sets.append(set(dates))

    common = sorted(set.intersection(*date_sets)) if date_sets else []
    if len(common) < 3:
        return None, None, None

    common = pd.to_datetime(common)
    ref_label = labels[0]
    obs_aligned = per_label[ref_label][1].reindex(common).values

    preds_aligned = {}
    for label in labels:
        preds_aligned[label] = per_label[label][2].reindex(common).values

    return common, obs_aligned, preds_aligned


def month_skill_heatmap(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    savepath: str
):
    """
    Create monthly skill heatmap.

    Args:
        dates: Date array
        y_true: Observed values
        y_pred: Predicted values
        title: Plot title
        savepath: Output file path
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": y_true, "yhat": y_pred}).dropna()
    if df.empty:
        return

    df["month"] = df["date"].dt.month
    agg = df.groupby("month").apply(
        lambda d: pd.Series({
            "RMSE": float(np.sqrt(((d.yhat - d.y) ** 2).mean())),
            "KGE": kge_2009(d.y.values, d.yhat.values),
            "R": float(np.corrcoef(d.y, d.yhat)[0, 1])
        }),
        include_groups=False
    ).reindex(range(1, 13))

    M = agg[["RMSE", "KGE", "R"]].values.astype(float)
    if not np.isfinite(M).any():
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(M.T, aspect="auto", cmap="RdYlGn")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["RMSE(↓)", "KGE(↑)", "R(↑)"])
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(m) for m in range(1, 13)])
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(savepath, dpi=SAVE_DPI)
    plt.close(fig)


def plot_permutation_importance(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: List[str],
    title: str,
    savepath: str,
    n_repeats: int = 10,
    logger: Optional[logging.Logger] = None
) -> Optional[List[str]]:
    """
    Plot permutation importance.

    Args:
        estimator: Fitted estimator
        X_test: Test features
        y_test: Test targets
        feature_names: Feature names
        title: Plot title
        savepath: Output file path
        n_repeats: Number of permutations
        logger: Logger instance

    Returns:
        List of top feature names
    """
    try:
        result = permutation_importance(
            estimator,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=DEFAULT_RANDOM_STATE,
            scoring=RMSE_SCORER
        )
    except Exception as e:
        if logger:
            logger.warning(f"Permutation importance failed: {e}")
        return None

    imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    imp[::-1].plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance (↑ more impactful)")
    hide_spines(ax)
    fig.tight_layout()
    fig.savefig(savepath, dpi=SAVE_DPI)
    plt.close(fig)

    return list(imp.index)


def plot_partial_dependence(
    estimator: Pipeline,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    top_features: List[str],
    title: str,
    savepath: str,
    logger: Optional[logging.Logger] = None
):
    """
    Plot partial dependence / ICE plots.

    Args:
        estimator: Fitted estimator
        X_sample: Sample of features
        feature_names: All feature names
        top_features: Features to plot
        title: Plot title
        savepath: Output file path
        logger: Logger instance
    """
    if not _HAS_PDP:
        if logger:
            logger.warning("PDP/ICE not available in this sklearn version")
        return

    feature_idx = [feature_names.index(f) for f in top_features if f in feature_names]
    if len(feature_idx) == 0:
        if logger:
            logger.warning("No valid features for PDP/ICE")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            PartialDependenceDisplay.from_estimator(
                estimator,
                X_sample,
                features=feature_idx,
                kind="both",
                ax=ax
            )
        except Exception:
            PartialDependenceDisplay.from_estimator(
                estimator,
                X_sample,
                features=feature_idx,
                kind="average",
                ax=ax
            )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(savepath, dpi=SAVE_DPI)
        plt.close(fig)
    except Exception as e:
        if logger:
            logger.warning(f"PDP/ICE plot failed: {e}")


def quality_report(
    label: str,
    df_precip: pd.DataFrame,
    spi_series: pd.Series,
    outdir: str,
    tag: str,
    logger: Optional[logging.Logger] = None
):
    """
    Generate data quality report.

    Args:
        label: Station label
        df_precip: Precipitation DataFrame
        spi_series: SPI series
        outdir: Output directory
        tag: File tag
        logger: Logger instance
    """
    pr = df_precip.copy()
    pr["precip"] = pd.to_numeric(pr["precip"], errors="coerce")

    miss_precip = float(pr["precip"].isna().mean())
    zeros_precip = float((pr["precip"] == 0).mean())

    spi = pd.to_numeric(spi_series, errors="coerce")
    miss_spi = float(pd.isna(spi).mean())

    if np.sum(~np.isnan(spi)) > 3:
        z = (spi - np.nanmean(spi)) / (np.nanstd(spi) + 1e-12)
        outlier_ratio = float(np.mean(np.abs(z) > 3))
        sp_skew = float(skew(spi[~np.isnan(spi)]))
        sp_kurt = float(kurtosis(spi[~np.isnan(spi)]))

        try:
            n_for_shapiro = min(5000, np.sum(~np.isnan(spi)))
            p_shapiro = float(shapiro(spi[~np.isnan(spi)][:n_for_shapiro]).pvalue)
        except Exception:
            p_shapiro = np.nan
    else:
        outlier_ratio = np.nan
        sp_skew = np.nan
        sp_kurt = np.nan
        p_shapiro = np.nan

    rep = pd.DataFrame([{
        "Station": label,
        "Missing_Precip": miss_precip,
        "Zeros_Precip": zeros_precip,
        "Missing_SPI": miss_spi,
        "Outlier_ratio_SPI(|z|>3)": outlier_ratio,
        "SPI_Skew": sp_skew,
        "SPI_Kurtosis": sp_kurt,
        "SPI_Shapiro_p": p_shapiro
    }])

    rep.to_csv(os.path.join(outdir, f"{tag}_quality_report.csv"), index=False)


# ==================== MAIN EVALUATION ====================
def evaluate_station(
    label: str,
    df_precip: pd.DataFrame,
    spi_scale: int,
    lead: int,
    config: ExperimentConfig,
    cache: CacheManager,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Evaluate single station for given SPI scale and lead time.

    Args:
        label: Station label
        df_precip: Precipitation DataFrame with [date, precip]
        spi_scale: SPI time scale in months
        lead: Forecast lead time in months
        config: Experiment configuration
        cache: Cache manager
        logger: Logger instance

    Returns:
        Dictionary with evaluation results
    """
    tag_base = f"{normalize_text(label)[:40]}_SPI{spi_scale}_LEAD{lead}"
    outdir = config.output_dir
    os.makedirs(outdir, exist_ok=True)

    if logger:
        logger.info(f"Evaluating: {label} | SPI-{spi_scale} | Lead={lead}")

    # Compute SPI
    cache_key = f"spi_{normalize_text(label)}_{spi_scale}"
    spi_df = cache.get(cache_key)

    if spi_df is None:
        spi_df = compute_spi(
            df_precip,
            value_col="precip",
            date_col="date",
            scale=spi_scale,
            calib_mode=config.spi_calibration_mode,
            calib_train_ratio=config.spi_calib_train_ratio,
            calib_start=config.spi_clim_start,
            calib_end=config.spi_clim_end,
            logger=logger
        )
        cache.set(cache_key, spi_df)

    data = df_precip[["date"]].merge(spi_df, on="date", how="left").rename(columns={f"spi_{spi_scale}": "y"})

    # Quality report
    if config.enable_quality_report:
        quality_report(label, df_precip[["date", "precip"]], data["y"], outdir, tag_base, logger)

    # Build variants
    variants = build_variant_frames(data[["date", "y"]].copy(), target_col="y", config=config, logger=logger)
    variants = apply_forecast_lead(variants, lead=lead)

    # Get reference dates
    any_variant = list(variants.keys())[0]
    vdf0 = variants[any_variant][["date", "y"]].dropna().reset_index(drop=True)
    y0 = vdf0["y"].values
    d0 = vdf0["date"].values
    split_idx = int(0.7 * len(vdf0))

    results = []
    preds_map = {}
    feature_names_map = {}
    best_estimators = {}
    best_params_list = []
    X_test_map = {}

    # Naive baselines (only for lead=0)
    if lead == 0 and split_idx >= 1 and len(vdf0) - split_idx >= 1:
        for naive_name, naive_lag in [("Naive-1", 1), ("Naive-12", 12)]:
            yhat = naive_forecast(y0, lag=naive_lag)[split_idx:]
            yte = y0[split_idx:]
            mask = np.isfinite(yhat) & np.isfinite(yte)

            if mask.sum() >= 3:
                mets = compute_all_metrics(yte[mask], yhat[mask])
                baseline_label = f"{naive_name} | NW | BASE"
                results.append({
                    "Label": baseline_label,
                    "Model": naive_name,
                    "Variant": "NW",
                    "Struct": "BASE",
                    "CV_RMSE": np.nan,
                    **mets
                })
                preds_map[baseline_label] = (d0[split_idx:], yte, yhat)
                feature_names_map[baseline_label] = []
                best_estimators[baseline_label] = None
                X_test_map[baseline_label] = None

                if logger:
                    logger.info(f"  {baseline_label:>30s}  RMSE={mets['RMSE']:.3f}  KGE={mets['KGE']:.3f}  R={mets['R']:.3f}")

    # Model evaluation loop
    enabled_models = [m for m, on in config.enabled_models.items() if on]

    for variant_name, variant_df in tqdm(
        variants.items(),
        desc=f"{label[:20]} variants",
        disable=not _HAS_TQDM or config.verbose < 1
    ):
        base_cols = [c for c in variant_df.columns if c not in ["date", "y"] and not c.startswith("lag") and not c.startswith("dnlag")]

        for struct_code, struct_lags in config.lag_structures.items():
            lag_cols = (
                [f"dnlag{L}" for L in struct_lags]
                if variant_name.startswith("WD")
                else [f"lag{L}" for L in struct_lags]
            )
            feat_cols = lag_cols + base_cols

            df_use = variant_df[["date", "y"] + feat_cols].dropna()
            if len(df_use) < max(config.min_train_samples, 3 * config.max_lag):
                continue

            X = df_use[feat_cols].values
            y = df_use["y"].values
            dates = df_use["date"].values

            split_local = int(0.7 * len(df_use))
            X_train, X_test = X[:split_local], X[split_local:]
            y_train, y_test = y[:split_local], y[split_local:]
            dates_test = dates[split_local:]

            for model_name in enabled_models:
                pipeline, param_space = get_model_pipeline(model_name, config)

                best_est, cv_rmse, best_params = fit_with_cv(
                    pipeline,
                    param_space,
                    X_train,
                    y_train,
                    config,
                    logger
                )

                y_pred = best_est.predict(X_test)
                mets = compute_all_metrics(y_test, y_pred)

                full_label = f"{model_name} | {variant_name} | {struct_code}"
                results.append({
                    "Label": full_label,
                    "Model": model_name,
                    "Variant": variant_name,
                    "Struct": struct_code,
                    "CV_RMSE": cv_rmse,
                    **mets
                })

                preds_map[full_label] = (dates_test, y_test, y_pred)
                feature_names_map[full_label] = feat_cols
                best_estimators[full_label] = best_est
                best_params_list.append({"Label": full_label, **best_params, "Features": ";".join(feat_cols)})
                X_test_map[full_label] = pd.DataFrame(X_test, columns=feat_cols)

                if logger:
                    logger.info(f"  {full_label:>30s}  RMSE={mets['RMSE']:.3f}  KGE={mets['KGE']:.3f}  R={mets['R']:.3f}")

    # Results analysis
    res_df = pd.DataFrame(results)
    if res_df.empty:
        raise RuntimeError(f"{label} (SPI{spi_scale}, L{lead}): No valid combinations generated")

    # Select top-K
    key = "KGE" if config.select_by.upper() == "KGE" else "RMSE"
    ascending = False if key == "KGE" else True
    topk_df = res_df.sort_values(key, ascending=ascending).head(config.top_k)
    top_labels = topk_df["Label"].tolist()

    # Best NW and best wavelet
    best_nw = (
        res_df[res_df["Variant"] == "NW"].sort_values(key, ascending=ascending).iloc[0]
        if "NW" in res_df["Variant"].unique()
        else topk_df.iloc[0]
    )
    non_nw = res_df[res_df["Variant"] != "NW"].sort_values(key, ascending=ascending)
    best_wave = non_nw.iloc[0] if not non_nw.empty else best_nw

    # Save results
    res_df.to_csv(os.path.join(outdir, f"{tag_base}_metrics_all.csv"), index=False)
    topk_df.set_index("Label").loc[top_labels, ["Model", "Variant", "Struct", "RMSE", "KGE", "R", "R2", "MAE", "NSE", "WD"]].round(4).to_csv(
        os.path.join(outdir, f"{tag_base}_top{config.top_k}_summary.csv")
    )
    if best_params_list:
        pd.DataFrame(best_params_list).to_csv(os.path.join(outdir, f"{tag_base}_best_params.csv"), index=False)

    # Visualizations
    if config.make_plots and len(top_labels) > 0:
        _create_visualizations(
            label,
            spi_scale,
            lead,
            top_labels,
            preds_map,
            topk_df,
            feature_names_map,
            best_estimators,
            X_test_map,
            config,
            outdir,
            tag_base,
            logger
        )

    return {
        "station": label,
        "scale": spi_scale,
        "lead": lead,
        "best_overall": topk_df.iloc[0].to_dict(),
        "best_nw": best_nw.to_dict(),
        "best_wave": best_wave.to_dict(),
        "topk": topk_df.copy()
    }


def _create_visualizations(
    label: str,
    spi_scale: int,
    lead: int,
    top_labels: List[str],
    preds_map: Dict[str, Tuple],
    topk_df: pd.DataFrame,
    feature_names_map: Dict[str, List[str]],
    best_estimators: Dict[str, Pipeline],
    X_test_map: Dict[str, pd.DataFrame],
    config: ExperimentConfig,
    outdir: str,
    tag_base: str,
    logger: Optional[logging.Logger] = None
):
    """Create all visualizations for top-K models."""
    # Time series overlay
    fig, ax = plt.subplots(figsize=(12, 4))
    d_ref, y_ref, _ = preds_map[top_labels[0]]
    ax.plot(d_ref, y_ref, label="Observation", lw=1.3, color="#333")

    colors = plt.cm.Set2(np.linspace(0, 1, len(top_labels)))
    for i, lbl in enumerate(top_labels):
        dates, y_true, y_pred = preds_map[lbl]
        ax.plot(dates, y_pred, label=lbl, lw=1.4, color=colors[i])

    year_axis(ax, step=5)
    ax.axhline(0, color="#777", lw=0.8)
    ax.set_title(f"{label} | SPI-{spi_scale} — Lead={lead} — Top {config.top_k}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"SPI-{spi_scale}")
    ax.legend(frameon=False, ncol=2)
    hide_spines(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_overlay.png"), dpi=SAVE_DPI)
    plt.close(fig)

    # Align predictions
    common_dates, obs_aligned, preds_aligned = align_predictions_by_common_dates(preds_map, top_labels)

    if common_dates is not None:
        # Taylor diagrams
        taylor_diagram_2d(
            obs_aligned,
            {lbl: preds_aligned[lbl] for lbl in top_labels},
            f"{label} — Taylor Diagram (Top {config.top_k}, SPI{spi_scale}, L{lead})",
            os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_taylor2d.png")
        )

        taylor_diagram_3d(
            obs_aligned,
            {lbl: preds_aligned[lbl] for lbl in top_labels},
            f"{label} — 3D Taylor (z=RMSD, Top {config.top_k}, SPI{spi_scale}, L{lead})",
            os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_taylor3d.png")
        )

        # Scatter plots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, lbl in enumerate(top_labels):
            ax = axes[i // 2, i % 2]
            yte = obs_aligned
            yhat = preds_aligned[lbl]
            ax.scatter(yte, yhat, s=18, alpha=0.85, color=colors[i], label=lbl)

            lims = [min(np.nanmin(yte), np.nanmin(yhat)), max(np.nanmax(yte), np.nanmax(yhat))]
            ax.plot(lims, lims, "k--", lw=1.0, label="1:1")

            try:
                k, b = np.polyfit(yte, yhat, 1)
                ax.plot(lims, [k * lims[0] + b, k * lims[1] + b], color="#444", lw=1.0, label=f"Reg: y={k:.2f}x+{b:.2f}")
            except Exception:
                pass

            row = topk_df[topk_df["Label"] == lbl].iloc[0]
            ax.set_title(f"{lbl}\nR={row['R']:.2f}, RMSE={row['RMSE']:.2f}, KGE={row['KGE']:.2f}")
            ax.set_xlabel("Observation")
            ax.set_ylabel("Prediction")
            ax.legend(frameon=False, fontsize=8)
            hide_spines(ax)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_scatter.png"), dpi=SAVE_DPI)
        plt.close(fig)

        # Residual series
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
        for i, lbl in enumerate(top_labels):
            ax = axes[i // 2, i % 2]
            dates = common_dates
            yte = obs_aligned
            yhat = preds_aligned[lbl]
            ax.plot(dates, yhat - yte, lw=1.1, color=colors[i])
            ax.axhline(0, color="#333", lw=1.0)
            year_axis(ax, step=5)
            ax.set_title(f"Residual — {lbl}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Residual")
            hide_spines(ax)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_residual_series.png"), dpi=SAVE_DPI)
        plt.close(fig)

        # Residual histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for i, lbl in enumerate(top_labels):
            ax = axes[i // 2, i % 2]
            r = preds_aligned[lbl] - obs_aligned
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue

            ax.hist(r, bins=30, density=True, alpha=0.45, color=colors[i], label="Hist")
            try:
                kde = sp_kde(r)
                xs = np.linspace(np.nanmin(r), np.nanmax(r), 200)
                ax.plot(xs, kde(xs), lw=1.5, color=colors[i], label="KDE")
            except Exception:
                pass

            ax.axvline(0, color="#333", lw=1.0)
            ax.set_title(f"Residual Distribution — {lbl}")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Density")
            hide_spines(ax)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_residual_hist.png"), dpi=SAVE_DPI)
        plt.close(fig)

        # QQ plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, lbl in enumerate(top_labels):
            ax = axes[i // 2, i % 2]
            r = preds_aligned[lbl] - obs_aligned
            r = r[np.isfinite(r)]
            if len(r) < 3:
                continue

            (osm, osr), (slope, intercept, r_val) = sp_probplot(r, dist="norm")
            ax.scatter(osm, osr, s=14, color=colors[i], alpha=0.85)
            ax.plot(osm, slope * osm + intercept, "k--", lw=1.0)
            ax.set_title(f"QQ-Plot — {lbl}")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            hide_spines(ax)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_qqplot.png"), dpi=SAVE_DPI)
        plt.close(fig)

        # Rolling metrics
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for i, lbl in enumerate(top_labels):
            dates = common_dates
            yte = obs_aligned
            yhat = preds_aligned[lbl]
            axes[0].plot(dates, rolling_metric(yte, yhat, ROLLING_WINDOW_MONTHS, "rmse"), lw=1.2, label=lbl, color=colors[i])

        axes[0].set_ylabel(f"RMSE (window={ROLLING_WINDOW_MONTHS})")
        axes[0].set_title("Rolling RMSE")
        axes[0].legend(frameon=False, ncol=2)
        hide_spines(axes[0])

        for i, lbl in enumerate(top_labels):
            dates = common_dates
            yte = obs_aligned
            yhat = preds_aligned[lbl]
            try:
                axes[1].plot(dates, rolling_metric(yte, yhat, ROLLING_WINDOW_MONTHS, "corr"), lw=1.2, label=lbl, color=colors[i])
            except Exception:
                pass

        axes[1].set_ylabel(f"Correlation (window={ROLLING_WINDOW_MONTHS})")
        axes[1].set_xlabel("Date")
        axes[1].set_title("Rolling Correlation")
        year_axis(axes[1], step=5)
        hide_spines(axes[1])
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_rolling_metrics.png"), dpi=SAVE_DPI)
        plt.close(fig)

    else:
        if logger:
            logger.warning("Insufficient common dates for aligned plots")

    # Metric bars
    mlist = top_labels
    rows = topk_df.set_index("Label").loc[mlist, ["RMSE", "KGE", "R2", "MAE", "NSE", "WD"]].astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    bcols = [plt.cm.Set2(i / (len(mlist) - 1) if len(mlist) > 1 else 0.5) for i in range(len(mlist))]
    (ax00, ax01, ax02), (ax10, ax11, ax12) = axes

    ax00.bar(mlist, rows["RMSE"], color=bcols)
    ax00.set_title("RMSE (↓)")
    set_xtick_rotation(ax00, 20, 'right')
    hide_spines(ax00)

    ax01.bar(mlist, rows["KGE"], color=bcols)
    ax01.set_title("KGE (↑)")
    ax01.set_ylim(-1, 1)
    set_xtick_rotation(ax01, 20, 'right')
    hide_spines(ax01)

    ax02.bar(mlist, rows["R2"], color=bcols)
    ax02.set_title("R² (↑)")
    ax02.set_ylim(0, 1)
    set_xtick_rotation(ax02, 20, 'right')
    hide_spines(ax02)

    ax10.bar(mlist, rows["MAE"], color=bcols)
    ax10.set_title("MAE (↓)")
    set_xtick_rotation(ax10, 20, 'right')
    hide_spines(ax10)

    ax11.bar(mlist, rows["NSE"], color=bcols)
    ax11.set_title("NSE (↑)")
    _nse = rows["NSE"].to_numpy(dtype=float)
    _nse = _nse[np.isfinite(_nse)]
    if _nse.size:
        bot = min(-1.0, float(_nse.min()) - 0.05 * abs(float(_nse.min())))
        bot = max(bot, -5.0)
        ax11.set_ylim(bot, 1.0)
    else:
        ax11.set_ylim(-1.0, 1.0)
    set_xtick_rotation(ax11, 20, 'right')
    hide_spines(ax11)

    ax12.bar(mlist, rows["WD"], color=bcols)
    ax12.set_title("Willmott d (↑)")
    ax12.set_ylim(0, 1)
    set_xtick_rotation(ax12, 20, 'right')
    hide_spines(ax12)

    fig.suptitle(f"Summary Metrics — Top {config.top_k} | SPI{spi_scale} | L{lead}", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag_base}_TOP{config.top_k}_metric_bars.png"), dpi=SAVE_DPI)
    plt.close(fig)

    # Optional diagnostics
    best_label = top_labels[0]
    dates_best, y_best, yhat_best = preds_map[best_label]

    if config.enable_month_skill_heatmap:
        month_skill_heatmap(
            dates_best,
            y_best,
            yhat_best,
            f"Monthly Skill — {label} (SPI{spi_scale}, L{lead})",
            os.path.join(outdir, f"{tag_base}_month_skill_heatmap.png")
        )

    if config.enable_perm_importance and X_test_map.get(best_label) is not None:
        top_feat_names = plot_permutation_importance(
            best_estimators[best_label],
            X_test_map[best_label],
            y_best,
            feature_names_map[best_label],
            f"Permutation Importance — {label} (Best, SPI{spi_scale}, L{lead})",
            os.path.join(outdir, f"{tag_base}_perm_importance.png"),
            logger=logger
        )
    else:
        top_feat_names = None

    if config.enable_pdp_ice and top_feat_names:
        pdp_feats = top_feat_names[:4]
        plot_partial_dependence(
            best_estimators[best_label],
            X_test_map[best_label],
            feature_names_map[best_label],
            pdp_feats,
            f"PDP/ICE — {label} (Best, SPI{spi_scale}, L{lead})",
            os.path.join(outdir, f"{tag_base}_pdp_ice.png"),
            logger
        )


# ==================== CLI ====================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wavelet-Enhanced SPI Forecasting System (Q1 Research Grade)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python wavelet_spi_forecasting_v2.py --input data.xlsx --output results/

  # Load YAML config
  python wavelet_spi_forecasting_v2.py --config myconfig.yaml

  # Fast mode
  python wavelet_spi_forecasting_v2.py --input data.xlsx --output results/ --fast

  # Single station
  python wavelet_spi_forecasting_v2.py --input data.xlsx --output results/ --station 17030

  # Multiple SPI scales and leads
  python wavelet_spi_forecasting_v2.py --input data.xlsx --output results/ --spi-scales 3 6 12 --leads 0 1 3 6
        """
    )

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--input", type=str, help="Input Excel file path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--station", type=str, help="Single station ID (omit for all stations)")
    parser.add_argument("--spi-scales", type=int, nargs="+", default=[12], help="SPI time scales (months)")
    parser.add_argument("--leads", type=int, nargs="+", default=[0], help="Forecast lead times (months)")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1, help="Verbosity level")
    parser.add_argument("--top-k", type=int, default=4, help="Number of top models to analyze")
    parser.add_argument("--select-by", type=str, choices=["KGE", "RMSE"], default="KGE", help="Model selection criterion")

    return parser.parse_args()


def create_summary_plots(
    summaries: List[Dict[str, Any]],
    outdir: str,
    logger: Optional[logging.Logger] = None
):
    """Create cross-station summary plots."""
    rows = []
    for s in summaries:
        def pick(d, fields=("RMSE", "KGE", "R2", "MAE", "NSE", "WD", "Variant", "Struct", "Model", "Label")):
            return {k: d.get(k, np.nan) for k in fields}

        best_all = pick(s["best_overall"])
        best_nw = pick(s["best_nw"])
        best_wv = pick(s["best_wave"])

        rows.append({
            "Station": s["station"],
            "SPI": s["scale"],
            "Lead": s["lead"],
            "BestAll_Model": best_all["Model"],
            "BestAll_Variant": best_all["Variant"],
            "BestAll_Struct": best_all["Struct"],
            "BestAll_RMSE": best_all["RMSE"],
            "BestAll_KGE": best_all["KGE"],
            "BestAll_R2": best_all["R2"],
            "BestAll_MAE": best_all["MAE"],
            "BestAll_NSE": best_all["NSE"],
            "BestAll_WD": best_all["WD"],
            "BestNW_Model": best_nw["Model"],
            "BestNW_Variant": best_nw["Variant"],
            "BestNW_Struct": best_nw["Struct"],
            "BestNW_RMSE": best_nw["RMSE"],
            "BestNW_KGE": best_nw["KGE"],
            "BestWV_Model": best_wv["Model"],
            "BestWV_Variant": best_wv["Variant"],
            "BestWV_Struct": best_wv["Struct"],
            "BestWV_RMSE": best_wv["RMSE"],
            "BestWV_KGE": best_wv["KGE"],
            "Delta_KGE": (best_wv["KGE"] - best_nw["KGE"]) if np.isfinite([best_wv["KGE"], best_nw["KGE"]]).all() else np.nan,
            "Delta_RMSE": (best_wv["RMSE"] - best_nw["RMSE"]) if np.isfinite([best_wv["RMSE"], best_nw["RMSE"]]).all() else np.nan,
        })

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(outdir, "ALL_stations_summary.csv"), index=False)

    if logger:
        logger.info("Saved ALL_stations_summary.csv")

    # Bar charts
    try:
        # KGE bar
        tmp = comp.sort_values("BestAll_KGE", ascending=False)
        fig, ax = plt.subplots(figsize=(max(9, len(tmp) * 0.35), 5))
        ax.bar(tmp["Station"] + "|S" + tmp["SPI"].astype(str) + "-L" + tmp["Lead"].astype(str), tmp["BestAll_KGE"])
        ax.set_title("Best Overall KGE — All Stations/SPI/Leads")
        ax.set_ylabel("KGE (↑)")
        set_xtick_rotation(ax, 60, 'right')
        hide_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ALL_best_KGE_bar.png"), dpi=SAVE_DPI)
        plt.close(fig)

        # RMSE bar
        tmp = comp.sort_values("BestAll_RMSE", ascending=True)
        fig, ax = plt.subplots(figsize=(max(9, len(tmp) * 0.35), 5))
        ax.bar(tmp["Station"] + "|S" + tmp["SPI"].astype(str) + "-L" + tmp["Lead"].astype(str), tmp["BestAll_RMSE"])
        ax.set_title("Best Overall RMSE — All Stations/SPI/Leads")
        ax.set_ylabel("RMSE (↓)")
        set_xtick_rotation(ax, 60, 'right')
        hide_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ALL_best_RMSE_bar.png"), dpi=SAVE_DPI)
        plt.close(fig)

        # Delta KGE
        fig, ax = plt.subplots(figsize=(max(9, len(comp) * 0.35), 5))
        ax.bar(comp["Station"] + "|S" + comp["SPI"].astype(str) + "-L" + comp["Lead"].astype(str), comp["Delta_KGE"])
        ax.axhline(0, color="#333", lw=1.0)
        ax.set_title("Wavelet Impact — ΔKGE (Wavelet − NW)")
        ax.set_ylabel("ΔKGE")
        set_xtick_rotation(ax, 60, 'right')
        hide_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ALL_delta_KGE_bar.png"), dpi=SAVE_DPI)
        plt.close(fig)

        # Delta RMSE
        fig, ax = plt.subplots(figsize=(max(9, len(comp) * 0.35), 5))
        ax.bar(comp["Station"] + "|S" + comp["SPI"].astype(str) + "-L" + comp["Lead"].astype(str), comp["Delta_RMSE"])
        ax.axhline(0, color="#333", lw=1.0)
        ax.set_title("Wavelet Impact — ΔRMSE (Wavelet − NW)")
        ax.set_ylabel("ΔRMSE (↓ better ≈ negative)")
        set_xtick_rotation(ax, 60, 'right')
        hide_spines(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ALL_delta_RMSE_bar.png"), dpi=SAVE_DPI)
        plt.close(fig)

    except Exception as e:
        if logger:
            logger.warning(f"Summary plot generation failed: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load or create config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        if not args.input or not args.output:
            print("ERROR: --input and --output required when not using --config")
            sys.exit(1)

        config = ExperimentConfig(
            input_file=args.input,
            output_dir=args.output,
            spi_scales=args.spi_scales,
            forecast_leads=args.leads,
            fast_mode=args.fast,
            run_all_stations=args.station is None,
            station_id=args.station,
            top_k=args.top_k,
            select_by=args.select_by,
            enable_caching=not args.no_cache,
            verbose=args.verbose
        )

    # Setup
    os.makedirs(config.output_dir, exist_ok=True)
    configure_matplotlib()
    logger = setup_logging(config.output_dir, config.verbose)
    cache = CacheManager(config.cache_dir, config.enable_caching)

    logger.info("=" * 70)
    logger.info("Wavelet-Enhanced SPI Forecasting System v2.0.0 (Q1 Ready)")
    logger.info("=" * 70)
    logger.info(f"Input: {config.input_file}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"SPI scales: {config.spi_scales}")
    logger.info(f"Leads: {config.forecast_leads}")
    logger.info(f"Fast mode: {config.fast_mode}")
    logger.info(f"Top-K: {config.top_k}")
    logger.info(f"Selection criterion: {config.select_by}")
    logger.info("=" * 70)

    # Save config
    config.to_yaml(os.path.join(config.output_dir, "experiment_config.yaml"))
    logger.info(f"Saved config to {os.path.join(config.output_dir, 'experiment_config.yaml')}")

    # Load stations
    stations, id_col = load_stations(config.input_file, config.id_column, logger)
    station_ids = list(stations.keys())

    logger.info(f"Loaded {len(station_ids)} stations from column '{id_col}'")

    # Select stations to process
    if config.run_all_stations:
        process_ids = station_ids
    else:
        if config.station_id not in stations:
            logger.warning(f"Station ID '{config.station_id}' not found, using first station")
            process_ids = [station_ids[0]]
        else:
            process_ids = [config.station_id]

    # Main evaluation loop
    summaries = []
    total_tasks = len(process_ids) * len(config.spi_scales) * len(config.forecast_leads)

    logger.info(f"Processing {len(process_ids)} stations × {len(config.spi_scales)} scales × {len(config.forecast_leads)} leads = {total_tasks} tasks")

    with tqdm(total=total_tasks, desc="Overall progress", disable=not _HAS_TQDM or config.verbose < 1) as pbar:
        for station_id in process_ids:
            station = stations[station_id]
            for spi_scale in config.spi_scales:
                for lead in config.forecast_leads:
                    try:
                        summary = evaluate_station(
                            station["label"],
                            station["df"],
                            spi_scale,
                            lead,
                            config,
                            cache,
                            logger
                        )
                        summaries.append(summary)
                    except Exception as e:
                        logger.error(f"Failed: {station['label']} | SPI{spi_scale} | L{lead}: {e}", exc_info=True)
                    finally:
                        pbar.update(1)

    # Cross-station summary
    if config.run_all_stations and summaries:
        logger.info("Creating cross-station summary")
        create_summary_plots(summaries, config.output_dir, logger)

    logger.info("=" * 70)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
