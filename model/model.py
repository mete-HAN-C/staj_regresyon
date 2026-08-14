import numpy as np
import joblib
from pathlib import Path
from load.load import drop_unwanted_columns, enforce_data_types

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "final_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_SELECTOR_PATH = MODEL_DIR / "feature_selector.pkl"


def save_artifacts(model, preprocessor, feature_selector):
    """Eğitilmiş model ve ön işleme nesnelerini diske kaydeder."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(feature_selector, FEATURE_SELECTOR_PATH)


def load_artifacts():
    """Diskten model ve ön işleme nesnelerini yükler."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_selector = joblib.load(FEATURE_SELECTOR_PATH)
    return model, preprocessor, feature_selector

def predict_new_data(raw_df):
    """
    Ham bir DataFrame alır (load_train/load_test ile okunmuş, hiç işlenmemiş),
    tüm pipeline'ı uygular ve SalePrice tahminlerini (gerçek dolar cinsinden) döndürür.
    """

    model, preprocessor, feature_selector = load_artifacts()

    df = drop_unwanted_columns(raw_df)
    df = enforce_data_types(df)
    df = preprocessor.transform(df)

    from clean.clean import drop_multicollinear
    df = drop_multicollinear(df)

    df = feature_selector.transform(df)

    predictions_log = model.predict(df)
    predictions = np.expm1(predictions_log)

    return predictions