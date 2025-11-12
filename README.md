# Wavelet-Enhanced SPI Forecasting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Grade](https://img.shields.io/badge/Status-Q1%20Ready-brightgreen.svg)]()

**Advanced drought forecasting using wavelet analysis and machine learning** — Publication-ready research code for hydrological modeling.

## 🌟 Features

### Core Capabilities
- **Multi-scale SPI computation** with robust statistical calibration (McKee et al., 1993)
- **Comprehensive wavelet analysis**:
  - Discrete Wavelet Transform (DWT) feature extraction
  - Wavelet-based denoising with MAD thresholding
  - Wavelet Packet Decomposition (WPD) for enhanced frequency analysis
- **Ensemble ML models**: HGBR, SVR, RF, ET, GBR, Ridge, KNN, LinearSVR
- **Automated hyperparameter tuning** with time-series cross-validation
- **Publication-quality visualizations**: Taylor diagrams, skill scores, residual analysis

### Research-Grade Engineering
- ✅ Type hints throughout for IDE support
- ✅ Professional logging with file + console output
- ✅ CLI interface with YAML configuration
- ✅ Intelligent caching for expensive computations
- ✅ Parallel processing with joblib
- ✅ Complete reproducibility tracking
- ✅ Leakage-proof time series validation

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/youruser/Sertac-Rev.git
cd Sertac-Rev

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Command-Line Usage

```bash
# Basic run (all stations, SPI-12)
python wavelet_spi_forecasting_v2.py \
    --input outputs/Stations_Thrace_Turker.xlsx \
    --output results/run1/

# Fast mode
python wavelet_spi_forecasting_v2.py \
    --input outputs/Stations_Thrace_Turker.xlsx \
    --output results/fast/ \
    --fast

# Multi-scale and multi-lead
python wavelet_spi_forecasting_v2.py \
    --input outputs/Stations_Thrace_Turker.xlsx \
    --output results/multi/ \
    --spi-scales 3 6 12 \
    --leads 0 1 3 6 12

# From YAML config
python wavelet_spi_forecasting_v2.py --config myconfig.yaml
```

## 📊 Output Structure

```
results/
├── experiment_config.yaml              # Saved configuration
├── run_20250112_143052.log            # Detailed log
├── ALL_stations_summary.csv           # Cross-station summary
├── StationXXX_SPI12_LEAD0_metrics_all.csv          # All results
├── StationXXX_SPI12_LEAD0_top4_summary.csv         # Top-4 models
├── StationXXX_SPI12_LEAD0_TOP4_taylor2d.png        # Visualizations
└── ... (more diagnostics)
```

## 📚 Citation

```bibtex
@software{wavelet_spi_2025,
  author = {Your Research Team},
  title = {Wavelet-Enhanced SPI Forecasting System},
  year = {2025},
  url = {https://github.com/youruser/Sertac-Rev}
}
```

## 📄 License

MIT License - See LICENSE file

---

**Version**: 2.0.0 | **Status**: ✅ Q1 Ready
