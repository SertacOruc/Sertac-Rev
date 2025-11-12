# Wavelet SPI Forecasting - Kullanım Kılavuzu

## 2 Versiyon Mevcut:

### 1. CLI İLE (wavelet_spi_WITH_CLI.py)
Komut satırından argüman alarak çalışır:

```bash
# Kullanım:
python wavelet_spi_WITH_CLI.py --input veri.xlsx --output sonuclar/

# Hızlı mod:
python wavelet_spi_WITH_CLI.py --input veri.xlsx --output sonuclar/ --fast

# Tek istasyon:
python wavelet_spi_WITH_CLI.py --input veri.xlsx --output sonuclar/ --station 17030

# YAML config ile:
python wavelet_spi_WITH_CLI.py --config ayarlar.yaml
```

### 2. CLI OLMADAN (wavelet_spi_NO_CLI.py)
Dosyanın içindeki ayarları düzenleyip direkt çalıştırırsınız:

```python
# Dosyada bu satırları bulup düzenleyin (satır ~30):
INPUT_FILE = "C:/veri/yagis.xlsx"       # ← Kendi dosya yolunuz
OUTPUT_DIR = "C:/sonuclar/"             # ← Çıktı klasörü
STATION_ID = None                       # ← None = tüm istasyonlar, "17030" = tek istasyon
```

Sonra direkt çalıştırın:
```bash
python wavelet_spi_NO_CLI.py
```

## Hangisini Kullanmalıyım?

- **Her seferinde farklı dosyalarla çalışacaksanız** → CLI versiyonu kullanın
- **Hep aynı dosyayla çalışacaksanız** → CLI'sız versiyon daha kolay

## Sorun Giderme

**Hata: "ERROR: --input and --output required"**
→ CLI versiyonunda argüman eksik, yukarıdaki örneklere bakın

**Hata: "File not found"**  
→ CLI'sız versiyonda `INPUT_FILE` yolunu kontrol edin

**Hata: "No valid combinations"**
→ Veri dosyanız formatına uygun değil, kolon isimlerini kontrol edin
