# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
Model Karşılaştırma Scripti
============================
Lasso, Ridge, ElasticNet, RandomForest, GradientBoosting ve XGBoost
modellerini aynı pipeline üzerinde eğitip Validation RMSE ile karşılaştırır.

Çalıştırmak için:
    python compare_models.py

Çıktılar:
    - Terminalde karşılaştırma tablosu
    - model_comparison.png  (bar grafiği — sunumda kullanılabilir)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlar için
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from load.load import (load_train, drop_unwanted_columns,
                       split_features_target, train_val_split, enforce_data_types)
from config.config import TARGET_COL
from clean.clean import (remove_outliers, FullPreprocessor,
                         drop_multicollinear, FeatureSelector)

# XGBoost isteğe bağlı
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[UYARI] XGBoost bulunamadi -> pip install xgboost")


# ─────────────────────────────────────────────────────────────────────────────
# 1) Veri Hazırlama (train.py ile birebir aynı pipeline)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Model Karşılaştırması Başlıyor")
print("=" * 60)

print("\n[1/3] Veri hazırlanıyor...")
df = load_train()
df = enforce_data_types(df)
df = remove_outliers(df, target_col=TARGET_COL)
df = drop_unwanted_columns(df)

X, y = split_features_target(df, target_col=TARGET_COL)
y_log = np.log1p(y)

X_train, X_val, y_train, y_val = train_val_split(X, y_log, test_size=0.2, random_state=42)

preprocessor = FullPreprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)

X_train_proc = drop_multicollinear(X_train_proc)
X_val_proc   = drop_multicollinear(X_val_proc)

feature_selector = FeatureSelector()
X_train_sel = feature_selector.fit_transform(X_train_proc, y_train)
X_val_sel   = feature_selector.transform(X_val_proc)

print(f"   Eğitim seti: {X_train_sel.shape} | Validasyon seti: {X_val_sel.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 2) Model Tanımları
# ─────────────────────────────────────────────────────────────────────────────
# Lasso için GridSearchCV'nin bulduğu en iyi alpha (0.0005) kullanılıyor.
# Diğer modeller için yaygın başlangıç hiperparametreleri seçildi.
MODELS = {
    "Lasso":            Lasso(alpha=0.0005, max_iter=10_000, random_state=42),
    "Ridge":            Ridge(alpha=10.0),
    "ElasticNet":       ElasticNet(alpha=0.0005, l1_ratio=0.5,
                                   max_iter=10_000, random_state=42),
    "RandomForest":     RandomForestRegressor(n_estimators=300,
                                              random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300,
                                                   learning_rate=0.05,
                                                   max_depth=4,
                                                   random_state=42),
}
if XGBOOST_AVAILABLE:
    MODELS["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        random_state=42, n_jobs=-1, verbosity=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3) Eğit & Değerlendir
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Modeller eğitiliyor ve değerlendiriliyor...")
print("-" * 60)

results = []

for name, model in MODELS.items():
    print(f"  ▶ {name:<20}", end="", flush=True)
    model.fit(X_train_sel, y_train)

    preds_log = model.predict(X_val_sel)
    rmse_log  = np.sqrt(mean_squared_error(y_val, preds_log))

    # Log RMSE'yi yaklaşık dolar hatasına çevir
    # median tahmin fiyatı üzerinden yüzdelik sapma olarak yorumlanır
    median_price = float(np.expm1(y_val.median()))
    approx_dollar_error = float(np.expm1(rmse_log)) * median_price

    results.append({
        "Model":            name,
        "RMSE (log)":       round(rmse_log, 4),
        "Yakl. Dolar Hata": int(approx_dollar_error),
    })
    print(f"  RMSE (log) = {rmse_log:.4f}  |  ~${approx_dollar_error:,.0f}")

print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Sonuç Tablosu
# ─────────────────────────────────────────────────────────────────────────────
df_results = (
    pd.DataFrame(results)
    .sort_values("RMSE (log)")
    .reset_index(drop=True)
)
df_results.index += 1  # 1'den başlasın

print("\nKARSILASTIRMA TABLOSU (RMSE'ye gore sirali)")
print("=" * 60)
print(df_results.to_string())
print("=" * 60)

best = df_results.iloc[0]
print(f"\nEn iyi model : {best['Model']}")
print(f"   RMSE (log) : {best['RMSE (log)']}")
print(f"   Yakl. hata : ~${best['Yakl. Dolar Hata']:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 5) Görsel — Bar Grafiği
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Grafik kaydediliyor -> model_comparison.png")

COLORS = {
    "Lasso":            "#4C72B0",
    "Ridge":            "#55A868",
    "ElasticNet":       "#C44E52",
    "RandomForest":     "#8172B2",
    "GradientBoosting": "#CCB974",
    "XGBoost":          "#64B5CD",
}

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(
    df_results["Model"],
    df_results["RMSE (log)"],
    color=[COLORS.get(m, "#999") for m in df_results["Model"]],
    edgecolor="white",
    height=0.55,
)

# Değer etiketleri
for bar, val in zip(bars, df_results["RMSE (log)"]):
    ax.text(
        val + 0.0005, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left", fontsize=10, fontweight="bold"
    )

# En iyi modeli vurgula
best_idx = df_results["RMSE (log)"].idxmin() - 1
bars[best_idx].set_edgecolor("gold")
bars[best_idx].set_linewidth(2.5)

ax.set_xlabel("Validation RMSE (log ölçeğinde) — düşük daha iyi", fontsize=11)
ax.set_title("Model Karşılaştırması — Validation RMSE", fontsize=13, fontweight="bold", pad=14)
ax.invert_yaxis()  # en iyi model en üstte
ax.set_xlim(0, df_results["RMSE (log)"].max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("   ✅ model_comparison.png kaydedildi.\n")
