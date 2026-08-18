import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

from load.load import (load_train, drop_unwanted_columns,
                       split_features_target, train_val_split, enforce_data_types)
from config.config import TARGET_COL
from clean.clean import (remove_outliers, FullPreprocessor,
                         drop_multicollinear, FeatureSelector)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[UYARI] XGBoost bulunamadi -> pip install xgboost")


print("=" * 60)
print("  Model Karşılaştırması Başlıyor")
print("=" * 60)

print("\n[1/4] Veri hazırlanıyor...")
df = load_train()
df = enforce_data_types(df)
df = remove_outliers(df, target_col=TARGET_COL)
df = drop_unwanted_columns(df)

X, y = split_features_target(df, target_col=TARGET_COL)
y_log = np.log1p(y)


X_train, X_val, y_train, y_val = train_val_split(X, y_log, test_size=0.2, random_state=42)

preprocessor = FullPreprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_train_proc = drop_multicollinear(X_train_proc)

feature_selector = FeatureSelector()
X_train_sel = feature_selector.fit_transform(X_train_proc, y_train)

print(f"   Hiperparametre arama seti: {X_train_sel.shape}")


BASE_MODELS = {
    "Lasso":            Lasso(max_iter=10_000, random_state=42),
    "Ridge":            Ridge(),
    "ElasticNet":       ElasticNet(max_iter=10_000, random_state=42),
    "RandomForest":     RandomForestRegressor(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}
if XGBOOST_AVAILABLE:
    BASE_MODELS["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)

PARAM_GRIDS = {
    "Lasso":    {"alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01]},
    "Ridge":    {"alpha": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]},
    "ElasticNet": {
        "alpha":    [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "RandomForest": {
        "n_estimators":    [200, 400],
        "max_depth":       [None, 10, 20],
        "min_samples_leaf": [1, 2],
    },
    "GradientBoosting": {
        "n_estimators":  [200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth":     [2, 3, 4],
    },
}
if XGBOOST_AVAILABLE:
    PARAM_GRIDS["XGBoost"] = {
        "n_estimators":  [200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth":     [2, 3, 4],
    }


print("\n[2/4] AŞAMA 1: Her model için GridSearchCV ile en iyi hiperparametreler aranıyor...")
print("-" * 60)

best_params = {}
for name, base_model in BASE_MODELS.items():
    print(f"   {name:<20}", end="", flush=True)
    grid = GridSearchCV(
        base_model, PARAM_GRIDS[name],
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train_sel, y_train)
    best_params[name] = grid.best_params_
    print(f"  en iyi params = {grid.best_params_}")

print("-" * 60)

print("\n[3/4] AŞAMA 2: Preprocessor + FeatureSelector + Model her fold'da BAŞTAN "
      "fit edilip 5-Fold CV ile karşılaştırılıyor...")
print("-" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, base_model in BASE_MODELS.items():
    print(f"   {name:<20}", end="", flush=True)
    fold_rmses = []

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        pp = FullPreprocessor()
        X_tr_p = pp.fit_transform(X_tr)
        X_va_p = pp.transform(X_va)
        X_tr_p = drop_multicollinear(X_tr_p)
        X_va_p = drop_multicollinear(X_va_p)

        fs = FeatureSelector()
        X_tr_s = fs.fit_transform(X_tr_p, y_tr)
        X_va_s = fs.transform(X_va_p)

        model = clone(base_model).set_params(**best_params[name])
        model.fit(X_tr_s, y_tr)

        preds = model.predict(X_va_s)
        fold_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))

    cv_mean = float(np.mean(fold_rmses))
    cv_std  = float(np.std(fold_rmses))

    median_price = float(np.expm1(y_log.median()))
    approx_dollar_error = float(np.expm1(cv_mean)) * median_price

    results.append({
        "Model":            name,
        "CV RMSE (log)":    round(cv_mean, 4),
        "CV Std":           round(cv_std, 4),
        "Yakl. Dolar Hata": int(approx_dollar_error),
    })
    print(f"  CV RMSE = {cv_mean:.4f} +/- {cv_std:.4f}  |  ~${approx_dollar_error:,.0f}")

print("-" * 60)



df_results = (
    pd.DataFrame(results)
    .sort_values("CV RMSE (log)")
    .reset_index(drop=True)
)
df_results.index += 1

print("\nKARSILASTIRMA TABLOSU (5-Fold CV RMSE'ye gore sirali)")
print("=" * 60)
print(df_results.to_string())
print("=" * 60)

winner = df_results.iloc[0]
print(f"\nONERILEN MODEL : {winner['Model']}")
print(f"   CV RMSE (log)          : {winner['CV RMSE (log)']} +/- {winner['CV Std']}")
print(f"   Yakl. dolar hata       : ~${winner['Yakl. Dolar Hata']:,}")
print(f"   En iyi hiperparametreler: {best_params[winner['Model']]}")

if len(df_results) > 1:
    second = df_results.iloc[1]
    diff = second["CV RMSE (log)"] - winner["CV RMSE (log)"]
    tolerance = max(winner["CV Std"], second["CV Std"])
    if diff < tolerance:
        print(f"\n   UYARI: {winner['Model']} ile {second['Model']} arasindaki fark "
              f"({diff:.4f}) fold-to-fold varyansin ({tolerance:.4f}) icinde kaliyor. "
              f"Yani aradaki ustunluk istatistiksel olarak guclu degil, ikisi de makul secim.")


print("\n[4/4] Grafik kaydediliyor -> model_comparison.png")

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
    df_results["CV RMSE (log)"],
    xerr=df_results["CV Std"],
    color=[COLORS.get(m, "#999") for m in df_results["Model"]],
    edgecolor="white",
    height=0.55,
    capsize=4,
    ecolor="#333333",
)

for bar, val, std in zip(bars, df_results["CV RMSE (log)"], df_results["CV Std"]):
    ax.text(
        val + std + 0.0005, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left", fontsize=10, fontweight="bold"
    )

ax.set_xlabel("5-Fold CV RMSE (log ölçeğinde, hata çubuğu = std) — düşük daha iyi", fontsize=11)
ax.set_title("Model Karşılaştırması — Tune Edilmiş, 5-Fold CV", fontsize=13, fontweight="bold", pad=14)
ax.invert_yaxis()
x_max = (df_results["CV RMSE (log)"] + df_results["CV Std"]).max() * 1.2
ax.set_xlim(0, x_max)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("    model_comparison.png kaydedildi.\n")