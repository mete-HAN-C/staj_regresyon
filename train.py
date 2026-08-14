import numpy as np
from sklearn.linear_model import Lasso
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
    
    print("\n9. Model eğitiliyor (Validation için)...")
    # GridSearchCV ile optimize edilen Lasso modeli entegre edildi
    model = Lasso(alpha=0.001, random_state=42)
    model.fit(X_train_selected, y_train)
    
    # Validation Seti üzerinde test
    val_preds = model.predict(X_val_selected)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"   => Validation RMSE (Log ölçeğinde): {rmse:.4f}")
    
    print("\n---------------------------------------------------")
    print("10. Final Modeli Tüm Veri Üzerinde Eğitiliyor...")
    # Genelde production için model, validation ve train birleşimiyle baştan eğitilir.
    
    preprocessor_final = FullPreprocessor()
    X_processed = preprocessor_final.fit_transform(X)
    X_processed = drop_multicollinear(X_processed)
    
    feature_selector_final = FeatureSelector()
    X_selected = feature_selector_final.fit_transform(X_processed, y_log)
    
    # Final model Lasso ile oluşturuldu
    final_model = Lasso(alpha=0.001, random_state=42)
    final_model.fit(X_selected, y_log)
    
    print("\n11. Model ve Artifacts (Preprocessor, FeatureSelector) diske kaydediliyor...")
    save_artifacts(final_model, preprocessor_final, feature_selector_final)
    print("\nEĞİTİM BAŞARIYLA TAMAMLANDI!")

if __name__ == "__main__":
    main()
