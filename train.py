import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

from load.load import load_train, drop_unwanted_columns, split_features_target, train_val_split, enforce_data_types
from config.config import TARGET_COL
from clean.clean import remove_outliers, FullPreprocessor, drop_multicollinear, FeatureSelector
from model.model import save_artifacts

def main():
    print("1. Veri yükleniyor...")
    df = load_train()

    df = enforce_data_types(df)

    print(f"   Başlangıç boyutu: {df.shape}")

    print("\n2. Aykırı değerler (Outliers) temizleniyor...")
    df = remove_outliers(df, target_col=TARGET_COL)
    print(f"   Aykırı değerler sonrası boyut: {df.shape}")

    print("\n3. Gereksiz kolonlar düşürülüyor...")
    df = drop_unwanted_columns(df)

    # X ve y ayrımı
    X, y = split_features_target(df, target_col=TARGET_COL)
    
    # 4. Hedef değişken log dönüşümü (Çok Önemli!)
    # model.py içindeki predict_new_data 'expm1' yaptığı için modeli 'log1p' üzerinde eğitmeliyiz.
    print("\n4. Hedef değişkene log1p dönüşümü uygulanıyor...")
    y_log = np.log1p(y)
    
    print("\n5. Eğitim ve doğrulama (Validation) setlerine ayrılıyor...")
    X_train, X_val, y_train, y_val = train_val_split(X, y_log, test_size=0.2, random_state=42)
    
    print("\n6. Ön işleme (Preprocessing) pipeline çalıştırılıyor...")
    preprocessor = FullPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    print("\n7. Çoklu doğrusal bağlantı (Multicollinearity) yapan kolonlar çıkarılıyor...")
    X_train_processed = drop_multicollinear(X_train_processed)
    X_val_processed = drop_multicollinear(X_val_processed)
    
    print("\n8. Özellik Seçimi (Feature Selection) yapılıyor...")
    feature_selector = FeatureSelector()
    X_train_selected = feature_selector.fit_transform(X_train_processed, y_train)
    X_val_selected = feature_selector.transform(X_val_processed)
    print(f"   Seçilen özellik sayısı: {X_train_selected.shape[1]}")
    
    print("\n9. GridSearchCV ile en iyi hiperparametreler aranıyor (ElasticNet)...")
    # ElasticNet'in iki hiperparametresi var:
    #   alpha   : regularization gücü (küçük = daha az ceza)
    #   l1_ratio: 0 → saf Ridge (L2), 1 → saf Lasso (L1), arası → ElasticNet
    param_grid = {
        "alpha":    [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    grid_search = GridSearchCV(
        ElasticNet(random_state=42, max_iter=10000),
        param_grid,
        cv=5,                        # 5-Fold cross-validation
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,                   # tüm CPU çekirdeklerini kullan
        verbose=1,
    )
    grid_search.fit(X_train_selected, y_train)

    best_alpha    = grid_search.best_params_["alpha"]
    best_l1_ratio = grid_search.best_params_["l1_ratio"]
    best_cv_rmse  = -grid_search.best_score_
    print(f"   => En iyi alpha    : {best_alpha}")
    print(f"   => En iyi l1_ratio : {best_l1_ratio}")
    print(f"   => En iyi CV RMSE  : {best_cv_rmse:.4f}")

    # Bulunan parametrelerle validation seti üzerinde son test
    model = grid_search.best_estimator_
    val_preds = model.predict(X_val_selected)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"   => Validation RMSE (Log ölçeğinde): {rmse:.4f}")
    
    print("\n10. K-Fold CV ile nihai model performansı değlendiriliyor...")
    # GridSearchCV adim 9'da hiperparametre seçimi için CV kullandı.
    # Bu adimda, bulunan en iyi parametrelerle tam pipeline (Preprocessor +
    # FeatureSelector + ElasticNet) 5-Fold CV ile baştan değlendiriliyor.
    # Her fold'da pipeline BASTAN fit ediliyor (data leakage onlemi).
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        pp = FullPreprocessor()
        X_tr = pp.fit_transform(X_tr)
        X_va = pp.transform(X_va)

        X_tr = drop_multicollinear(X_tr)
        X_va = drop_multicollinear(X_va)

        fs = FeatureSelector()
        X_tr = fs.fit_transform(X_tr, y_tr)
        X_va = fs.transform(X_va)

        m = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                       random_state=42, max_iter=10000)
        m.fit(X_tr, y_tr)

        fold_rmse = np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
        cv_scores.append(fold_rmse)
        print(f"   Fold {fold}: RMSE = {fold_rmse:.4f}")

    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)
    print(f"   => 5-Fold CV RMSE : {cv_mean:.4f} +/- {cv_std:.4f}")

    print("\n---------------------------------------------------")
    print("11. Final Modeli Tüm Veri Üzerinde Eğitiliyor...")
    # Genelde production için model, validation ve train birleşimiyle baştan eğitilir.
    
    preprocessor_final = FullPreprocessor()
    X_processed = preprocessor_final.fit_transform(X)
    X_processed = drop_multicollinear(X_processed)
    
    feature_selector_final = FeatureSelector()
    X_selected = feature_selector_final.fit_transform(X_processed, y_log)
    
    # Final model: GridSearchCV'nin bulduğu en iyi alpha + l1_ratio ile
    final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                             random_state=42, max_iter=10000)
    final_model.fit(X_selected, y_log)
    
    print("\n12. Model ve Artifacts (Preprocessor, FeatureSelector) diske kaydediliyor...")
    save_artifacts(final_model, preprocessor_final, feature_selector_final)
    print("\nEĞİTİM BAŞARIYLA TAMAMLANDI!")

if __name__ == "__main__":
    main()
