"""
HDD Day-Ahead Tahmin - ML + Metaheuristic Optimizasyon
======================================================
Modeller: SVR, RF, XGBoost, MLP
Metaheuristics: GWO, GA, DE
Features: lag1, lag2, lag3, lag5, lag8
Metrik: RMSE (minimize)
CV: TimeSeriesSplit (5-fold)

Kullanım:
    python hdd_forecast_ml.py                    # Fiktif veri ile çalışır
    python hdd_forecast_ml.py --csv yol/dosya.csv  # Gerçek veri ile çalışır
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
import argparse
import os

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# 0. FİKTİF VERİ OLUŞTURMA
# =============================================================================
def generate_synthetic_hdd_data(n_days=730, seed=42):
    """
    Gerçekçi HDD (Heating Degree Days) verisi oluşturur.

    HDD formülü: max(0, T_base - T_avg)
    - Kışın yüksek (soğuk günler, ısıtma gerekli)
    - Yazın düşük veya sıfır (sıcak günler)

    Parameters:
        n_days: Gün sayısı (default: 730 = 2 yıl)
        seed: Random seed

    Returns:
        DataFrame with Date and HDD columns
    """
    np.random.seed(seed)

    # Tarih aralığı oluştur
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

    # Yılın günü (1-365) - numpy array'e çevir
    day_of_year = dates.dayofyear.values

    # Mevsimsel pattern: Kış yüksek, yaz düşük
    # Sinüs fonksiyonu ile mevsimsellik (Ocak=max, Temmuz=min)
    # Faz kaydırması: Ocak ortası maksimum olsun
    seasonal = 8 * np.cos(2 * np.pi * (day_of_year - 15) / 365)  # -8 ile +8 arası

    # Baz HDD değeri (ortalama)
    base_hdd = 7

    # Trend (hafif azalma - iklim değişikliği etkisi simülasyonu)
    trend = -0.001 * np.arange(n_days)

    # Rastgele gürültü (autoregressive component için)
    noise = np.zeros(n_days)
    noise[0] = np.random.normal(0, 1)
    for i in range(1, n_days):
        # AR(1) süreci: bugünkü gürültü dünküne bağlı
        noise[i] = 0.3 * noise[i-1] + np.random.normal(0, 1.5)

    # HDD hesapla
    hdd = base_hdd + seasonal + trend + noise

    # HDD negatif olamaz
    hdd = np.maximum(hdd, 0)

    # Yaz aylarında düşük değerler (Haziran-Ağustos)
    summer_mask = ((day_of_year >= 152) & (day_of_year <= 243))  # ~1 Haziran - 31 Ağustos
    hdd[summer_mask] = hdd[summer_mask] * 0.1 + np.random.uniform(0, 0.5, summer_mask.sum())

    # Yuvarlama
    hdd = np.round(hdd, 1)

    df = pd.DataFrame({
        'Date': dates,
        'HDD': hdd
    })

    return df


def load_csv_data(csv_path):
    """
    CSV dosyasından HDD verisi yükler.

    Desteklenen formatlar:
    - Noktalı virgül (;) ayraçlı, virgül (,) ondalık
    - Virgül (,) ayraçlı, nokta (.) ondalık
    """
    # Dosya var mı kontrol et
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")

    # Önce noktalı virgül ayraçlı dene
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',', na_values=['-', ''])
        if df.shape[1] < 2:
            raise ValueError("Tek sütun")
    except:
        # Virgül ayraçlı dene
        df = pd.read_csv(csv_path, na_values=['-', ''])

    # Sütun isimlerini kontrol et
    print(f"CSV sütunları: {df.columns.tolist()}")

    # Tarih ve HDD sütunlarını bul
    date_col = None
    hdd_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower or 'date' in col_lower or 'tarih' in col_lower:
            date_col = col
        if 'hdd' in col_lower or 'heating' in col_lower or 'degree' in col_lower:
            hdd_col = col

    if date_col is None:
        date_col = df.columns[0]
        print(f"Tarih sütunu bulunamadı, ilk sütun kullanılıyor: {date_col}")

    if hdd_col is None:
        # Sayısal sütunlardan birini seç
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            hdd_col = numeric_cols[0]
        else:
            hdd_col = df.columns[1]
        print(f"HDD sütunu bulunamadı, şu sütun kullanılıyor: {hdd_col}")

    df = df[[date_col, hdd_col]].copy()
    df.columns = ['Date', 'HDD']

    # Tarih dönüşümü
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    except:
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        except:
            df['Date'] = pd.to_datetime(df['Date'])

    df['HDD'] = pd.to_numeric(df['HDD'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    return df


# =============================================================================
# 1. VERİ HAZIRLAMA
# =============================================================================
def prepare_data(df):
    """Feature engineering ve veri hazırlama"""
    # Feature'lar: lag1, lag2, lag3, lag5, lag8
    for lag in [1, 2, 3, 5, 8]:
        df[f'lag{lag}'] = df['HDD'].shift(lag)

    df = df.dropna().reset_index(drop=True)

    feature_cols = ['lag1', 'lag2', 'lag3', 'lag5', 'lag8']
    X = df[feature_cols].values
    y = df['HDD'].values

    # Standardizasyon
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, scaler_X, scaler_y, feature_cols, df


# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
# =============================================================================
def evaluate_model(model, X, y, cv):
    """Model performansını RMSE olarak değerlendir"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    return -scores.mean(), scores.std()


def decode_params_svr(individual):
    """SVR parametrelerini decode et"""
    C = 10 ** (individual[0] * 4 - 1)  # [0.1, 1000] log-scale
    epsilon = 10 ** (individual[1] * 2.7 - 3)  # [0.001, 0.5] log-scale
    gamma = 10 ** (individual[2] * 3.7 - 3)  # [0.001, 5] log-scale
    return {'C': C, 'epsilon': epsilon, 'gamma': gamma}


def decode_params_rf(individual):
    """RF parametrelerini decode et"""
    n_estimators = int(individual[0] * 250 + 50)  # [50, 300]
    max_depth = int(individual[1] * 17 + 3)  # [3, 20]
    min_samples_split = int(individual[2] * 13 + 2)  # [2, 15]
    return {'n_estimators': n_estimators, 'max_depth': max_depth,
            'min_samples_split': min_samples_split, 'random_state': 42, 'n_jobs': -1}


def decode_params_xgb(individual):
    """XGBoost parametrelerini decode et"""
    n_estimators = int(individual[0] * 250 + 50)  # [50, 300]
    max_depth = int(individual[1] * 9 + 3)  # [3, 12]
    learning_rate = individual[2] * 0.29 + 0.01  # [0.01, 0.3]
    subsample = individual[3] * 0.4 + 0.6  # [0.6, 1.0]
    return {'n_estimators': n_estimators, 'max_depth': max_depth,
            'learning_rate': learning_rate, 'subsample': subsample,
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0}


def decode_params_mlp(individual):
    """MLP parametrelerini decode et"""
    layer1 = int(individual[0] * 96 + 32)  # [32, 128]
    layer2 = int(individual[1] * 48 + 16)  # [16, 64]
    alpha = 10 ** (individual[2] * 3 - 4)  # [0.0001, 0.1] log-scale
    lr = 10 ** (individual[3] * 2 - 4)  # [0.0001, 0.01] log-scale
    return {'hidden_layer_sizes': (layer1, layer2), 'alpha': alpha,
            'learning_rate_init': lr, 'max_iter': 500, 'random_state': 42,
            'early_stopping': True, 'n_iter_no_change': 20}


def create_model(model_name, params):
    """Model oluştur"""
    if model_name == 'SVR':
        return SVR(kernel='rbf', **params)
    elif model_name == 'RF':
        return RandomForestRegressor(**params)
    elif model_name == 'XGB':
        return xgb.XGBRegressor(**params)
    elif model_name == 'MLP':
        return MLPRegressor(**params)


def objective_function(individual, model_name, X, y, cv):
    """Fitness fonksiyonu - RMSE döndürür"""
    try:
        if model_name == 'SVR':
            params = decode_params_svr(individual)
        elif model_name == 'RF':
            params = decode_params_rf(individual)
        elif model_name == 'XGB':
            params = decode_params_xgb(individual)
        elif model_name == 'MLP':
            params = decode_params_mlp(individual)

        model = create_model(model_name, params)
        rmse, _ = evaluate_model(model, X, y, cv)
        return rmse
    except:
        return 999.0  # Hata durumunda yüksek değer


# =============================================================================
# 3. METAHEURİSTİK ALGORİTMALAR
# =============================================================================
def gwo_optimize(objective, dim, bounds, pop_size=30, max_iter=50, model_name='', X=None, y=None, cv=None):
    """Grey Wolf Optimizer"""
    # Başlangıç popülasyonu
    wolves = np.random.rand(pop_size, dim)

    # Fitness hesapla
    fitness = np.array([objective(w, model_name, X, y, cv) for w in wolves])

    # Alpha, Beta, Delta
    sorted_idx = np.argsort(fitness)
    alpha = wolves[sorted_idx[0]].copy()
    beta = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()
    alpha_score = fitness[sorted_idx[0]]
    beta_score = fitness[sorted_idx[1]]
    delta_score = fitness[sorted_idx[2]]

    for iteration in range(max_iter):
        a = 2 - iteration * (2 / max_iter)  # a: 2 -> 0

        for i in range(pop_size):
            for j in range(dim):
                # Alpha
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A1 * D_alpha

                # Beta
                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i, j])
                X2 = beta[j] - A2 * D_beta

                # Delta
                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i, j])
                X3 = delta[j] - A3 * D_delta

                wolves[i, j] = np.clip((X1 + X2 + X3) / 3, 0, 1)

        # Fitness güncelle
        fitness = np.array([objective(w, model_name, X, y, cv) for w in wolves])

        # Liderler güncelle
        for i in range(pop_size):
            if fitness[i] < alpha_score:
                delta, delta_score = beta.copy(), beta_score
                beta, beta_score = alpha.copy(), alpha_score
                alpha, alpha_score = wolves[i].copy(), fitness[i]
            elif fitness[i] < beta_score:
                delta, delta_score = beta.copy(), beta_score
                beta, beta_score = wolves[i].copy(), fitness[i]
            elif fitness[i] < delta_score:
                delta, delta_score = wolves[i].copy(), fitness[i]

    return alpha, alpha_score


def ga_optimize(objective, dim, bounds, pop_size=30, max_iter=50, model_name='', X=None, y=None, cv=None):
    """Genetic Algorithm"""
    # Başlangıç popülasyonu
    population = np.random.rand(pop_size, dim)
    fitness = np.array([objective(ind, model_name, X, y, cv) for ind in population])

    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    mutation_rate = 0.1
    crossover_rate = 0.8

    for iteration in range(max_iter):
        # Selection (Tournament)
        new_population = []
        for _ in range(pop_size):
            idx1, idx2 = np.random.randint(0, pop_size, 2)
            winner = idx1 if fitness[idx1] < fitness[idx2] else idx2
            new_population.append(population[winner].copy())

        # Crossover
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < crossover_rate:
                cx_point = np.random.randint(1, dim)
                new_population[i][cx_point:], new_population[i + 1][cx_point:] = \
                    new_population[i + 1][cx_point:].copy(), new_population[i][cx_point:].copy()

        # Mutation
        for i in range(pop_size):
            for j in range(dim):
                if np.random.rand() < mutation_rate:
                    new_population[i][j] = np.clip(new_population[i][j] + np.random.randn() * 0.1, 0, 1)

        population = np.array(new_population)
        fitness = np.array([objective(ind, model_name, X, y, cv) for ind in population])

        # Best güncelle
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_individual = population[min_idx].copy()
            best_fitness = fitness[min_idx]

        # Elitism - en iyiyi koru
        worst_idx = np.argmax(fitness)
        population[worst_idx] = best_individual.copy()
        fitness[worst_idx] = best_fitness

    return best_individual, best_fitness


def de_optimize(objective, dim, bounds, pop_size=30, max_iter=50, model_name='', X=None, y=None, cv=None):
    """Differential Evolution"""
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover rate

    # Başlangıç popülasyonu
    population = np.random.rand(pop_size, dim)
    fitness = np.array([objective(ind, model_name, X, y, cv) for ind in population])

    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    for iteration in range(max_iter):
        for i in range(pop_size):
            # 3 farklı birey seç
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # Mutant vektör
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = np.clip(mutant, 0, 1)

            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Selection
            trial_fitness = objective(trial, model_name, X, y, cv)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_individual = trial.copy()
                    best_fitness = trial_fitness

    return best_individual, best_fitness


# =============================================================================
# 4. ANA ÇALIŞMA FONKSİYONU
# =============================================================================
def run_optimization(X_scaled, y_scaled, tscv, pop_size=30, max_iter=50):
    """Tüm modeller için baseline ve optimizasyon çalıştır"""

    # --- BASELINE MODELLER ---
    print("\n" + "=" * 70)
    print("BASELINE MODELLER (Default Parametreler)")
    print("=" * 70)

    baseline_results = {}

    # SVR Baseline
    svr_default = SVR(kernel='rbf')
    rmse, std = evaluate_model(svr_default, X_scaled, y_scaled, tscv)
    baseline_results['SVR'] = {'RMSE': rmse, 'Std': std, 'Params': 'default'}
    print(f"SVR:     RMSE = {rmse:.4f} +/- {std:.4f}")

    # RF Baseline
    rf_default = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rmse, std = evaluate_model(rf_default, X_scaled, y_scaled, tscv)
    baseline_results['RF'] = {'RMSE': rmse, 'Std': std, 'Params': 'default'}
    print(f"RF:      RMSE = {rmse:.4f} +/- {std:.4f}")

    # XGBoost Baseline
    xgb_default = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    rmse, std = evaluate_model(xgb_default, X_scaled, y_scaled, tscv)
    baseline_results['XGB'] = {'RMSE': rmse, 'Std': std, 'Params': 'default'}
    print(f"XGB:     RMSE = {rmse:.4f} +/- {std:.4f}")

    # MLP Baseline
    mlp_default = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True)
    rmse, std = evaluate_model(mlp_default, X_scaled, y_scaled, tscv)
    baseline_results['MLP'] = {'RMSE': rmse, 'Std': std, 'Params': 'default'}
    print(f"MLP:     RMSE = {rmse:.4f} +/- {std:.4f}")

    # --- METAHEURİSTİK OPTİMİZASYON ---
    print("\n" + "=" * 70)
    print(f"METAHEURİSTİK OPTİMİZASYON (Pop={pop_size}, Iter={max_iter})")
    print("=" * 70)

    # Model ve boyut bilgileri
    model_dims = {'SVR': 3, 'RF': 3, 'XGB': 4, 'MLP': 4}
    metaheuristics = {
        'GWO': gwo_optimize,
        'GA': ga_optimize,
        'DE': de_optimize
    }

    all_results = []

    for model_name in ['SVR', 'RF', 'XGB', 'MLP']:
        dim = model_dims[model_name]
        print(f"\n--- {model_name} ---")

        for meta_name, meta_func in metaheuristics.items():
            print(f"  {meta_name} calisiyor...", end=" ", flush=True)

            best_individual, best_rmse = meta_func(
                objective=objective_function,
                dim=dim,
                bounds=None,
                pop_size=pop_size,
                max_iter=max_iter,
                model_name=model_name,
                X=X_scaled,
                y=y_scaled,
                cv=tscv
            )

            # En iyi parametreleri decode et
            if model_name == 'SVR':
                best_params = decode_params_svr(best_individual)
            elif model_name == 'RF':
                best_params = decode_params_rf(best_individual)
            elif model_name == 'XGB':
                best_params = decode_params_xgb(best_individual)
            elif model_name == 'MLP':
                best_params = decode_params_mlp(best_individual)

            # Son doğrulama
            model = create_model(model_name, best_params)
            final_rmse, final_std = evaluate_model(model, X_scaled, y_scaled, tscv)

            print(f"RMSE = {final_rmse:.4f} +/- {final_std:.4f}")

            all_results.append({
                'Model': model_name,
                'Optimizer': meta_name,
                'RMSE': final_rmse,
                'Std': final_std,
                'Params': str(best_params)
            })

    # Baseline sonuçlarını ekle
    for model_name, data in baseline_results.items():
        all_results.append({
            'Model': model_name,
            'Optimizer': 'Baseline',
            'RMSE': data['RMSE'],
            'Std': data['Std'],
            'Params': data['Params']
        })

    return all_results, baseline_results


def print_results(all_results, baseline_results):
    """Sonuçları yazdır ve kaydet"""

    print("\n" + "=" * 70)
    print("SONUC TABLOSU")
    print("=" * 70)

    # DataFrame oluştur
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['Model', 'RMSE'])

    print("\n" + results_df.to_string(index=False))

    # En iyi sonuçlar
    print("\n" + "=" * 70)
    print("EN IYI SONUCLAR (Her Model Icin)")
    print("=" * 70)

    for model_name in ['SVR', 'RF', 'XGB', 'MLP']:
        model_df = results_df[results_df['Model'] == model_name]
        best = model_df.loc[model_df['RMSE'].idxmin()]
        baseline = model_df[model_df['Optimizer'] == 'Baseline'].iloc[0]
        improvement = (baseline['RMSE'] - best['RMSE']) / baseline['RMSE'] * 100

        print(f"\n{model_name}:")
        print(f"  Baseline:  RMSE = {baseline['RMSE']:.4f}")
        print(f"  En Iyi:    RMSE = {best['RMSE']:.4f} ({best['Optimizer']})")
        print(f"  Iyilesme:  {improvement:.2f}%")

    # Genel en iyi
    print("\n" + "=" * 70)
    print("GENEL EN IYI MODEL")
    print("=" * 70)
    best_overall = results_df.loc[results_df['RMSE'].idxmin()]
    print(f"Model: {best_overall['Model']}")
    print(f"Optimizer: {best_overall['Optimizer']}")
    print(f"RMSE: {best_overall['RMSE']:.4f} +/- {best_overall['Std']:.4f}")
    print(f"Params: {best_overall['Params']}")

    # CSV kaydet
    output_file = 'ml_optimization_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSonuclar kaydedildi: {output_file}")

    return results_df


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='HDD Day-Ahead Tahmin - ML + Metaheuristic Optimizasyon')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV dosya yolu (verilmezse fiktif veri kullanilir)')
    parser.add_argument('--synthetic-days', type=int, default=730,
                        help='Fiktif veri icin gun sayisi (default: 730)')
    parser.add_argument('--pop-size', type=int, default=30,
                        help='Metaheuristic populasyon boyutu (default: 30)')
    parser.add_argument('--max-iter', type=int, default=50,
                        help='Metaheuristic iterasyon sayisi (default: 50)')

    args = parser.parse_args()

    print("=" * 70)
    print("HDD DAY-AHEAD TAHMIN - ML + METAHEURISTIC OPTIMIZASYON")
    print("=" * 70)

    # Veri yükleme
    print("\n" + "=" * 70)
    print("VERI HAZIRLAMA")
    print("=" * 70)

    if args.csv:
        print(f"CSV dosyasi yukleniyor: {args.csv}")
        try:
            df = load_csv_data(args.csv)
            print(f"Basariyla yuklendi: {len(df)} satir")
        except FileNotFoundError as e:
            print(f"HATA: {e}")
            print("Fiktif veri kullaniliyor...")
            df = generate_synthetic_hdd_data(args.synthetic_days)
    else:
        print(f"Fiktif veri olusturuluyor ({args.synthetic_days} gun)...")
        df = generate_synthetic_hdd_data(args.synthetic_days)
        print("Fiktif veri ornegi:")
        print(df.head(10).to_string(index=False))

    # Feature engineering
    X_scaled, y_scaled, scaler_X, scaler_y, feature_cols, df_processed = prepare_data(df)

    print(f"\nVeri boyutu: {X_scaled.shape[0]} gun, {X_scaled.shape[1]} feature")
    print(f"Features: {feature_cols}")
    print(f"HDD istatistikleri:")
    print(f"  Min: {df_processed['HDD'].min():.2f}")
    print(f"  Max: {df_processed['HDD'].max():.2f}")
    print(f"  Mean: {df_processed['HDD'].mean():.2f}")
    print(f"  Std: {df_processed['HDD'].std():.2f}")

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Optimizasyon çalıştır
    all_results, baseline_results = run_optimization(
        X_scaled, y_scaled, tscv,
        pop_size=args.pop_size,
        max_iter=args.max_iter
    )

    # Sonuçları yazdır
    results_df = print_results(all_results, baseline_results)

    return results_df


if __name__ == '__main__':
    results = main()
