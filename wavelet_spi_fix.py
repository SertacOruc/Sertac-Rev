#!/usr/bin/env python3
"""
Quick fix for CLI argument validation
"""
import sys
import argparse

def parse_args_fixed():
    parser = argparse.ArgumentParser(
        description="Wavelet-Enhanced SPI Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py --input data.xlsx --output results/
  python script.py --config config.yaml
        """
    )
    
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--input", help="Input Excel file")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    # FIXED: Show help if required args missing
    if not args.config and (not args.input or not args.output):
        parser.print_help()
        print("\n" + "="*60)
        print("ERROR: Missing required arguments!")
        print("="*60)
        print("\nProvide either:")
        print("  --config <file>  OR  --input <file> --output <dir>")
        print("="*60)
        sys.exit(1)
    
    return args

if __name__ == "__main__":
    args = parse_args_fixed()
    print("Arguments OK!")
