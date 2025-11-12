# KENDİ DOSYANIZDA YAPMANIZ GEREKEN TEK DEĞİŞİKLİK

## SORUN:
Kodunuzda `parse_args()` fonksiyonunda **ulaşılamaz kod** hatası var.

## ÇÖZÜM:

### Adım 1: Dosyayı açın
`wavelet_spi_forecasting_v2.py` dosyanızı text editörde açın

### Adım 2: parse_args() fonksiyonunu bulun (yaklaşık satır 3630)

### Adım 3: ŞU SATIRLARI BULUN:
```python
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(...)
    
    # ... tüm argument tanımlamaları ...
    
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
    
    return args  # ✅ BU SATIR DOĞRU - TUTUN

    return parser.parse_args()  # ❌ BU SATIRI SİLİN!
```

### Adım 4: Son satırı silin
`return parser.parse_args()` satırını **TAM OLARAK SİLİN**

### SONUÇ:
```python
def parse_args():
    # ... tüm kod aynı ...
    
    return args  # ✅ Sadece bu return kalacak
    # ❌ İkinci return satırı silinmiş olmalı
```

### Adım 5: Dosyayı kaydedin ve test edin
```bash
python wavelet_spi_forecasting_v2.py --input veri.xlsx --output sonuc/
```

## EĞER HÂLÂ ÇALIŞMAZSA:

Repo'daki **ÇALIŞAN** dosyaları kullanın:
- `wavelet_spi_WITH_CLI.py` - CLI ile (komut satırı argümanları)  
- `wavelet_spi_NO_CLI.py` - CLI'sız (hard-coded ayarlar)

