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
Version: 2.0.1 (Q1 Ready - Fixed CLI)
License: MIT
"""

# ... (dosyanın geri kalanı aynı - sadece parse_args değişecek)
# Tüm içeriği kopyalamak yerine, sadece değişikliği gösteriyorum

import sys
import argparse

# ... tüm imports ve kod aynı ...

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

    args = parser.parse_args()

    # ✅ VALIDATION: Show help if required arguments are missing
    if not args.config and (not args.input or not args.output):
        parser.print_help()
        print("\n" + "="*70)
        print("ERROR: Missing required arguments!")
        print("="*70)
        print("\nYou must provide either:")
        print("  1. --config <yaml_file>")
        print("     OR")
        print("  2. --input <data_file> --output <output_directory>")
        print("\nExamples:")
        print("  python wavelet_spi_forecasting_v2.py --input data.xlsx --output results/")
        print("  python wavelet_spi_forecasting_v2.py --config config.yaml")
        print("="*70)
        sys.exit(1)

    return args
