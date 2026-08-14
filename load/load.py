import pandas as pd
from config.config import TRAIN_PATH, TEST_PATH, DROP_COLS, TARGET_COL
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def load_train():
    return load_data(TRAIN_PATH)

def load_test():
    return load_data(TEST_PATH)

def drop_unwanted_columns(df):
    df = df.copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)

# MoSold (Satış Ayı) eklendi
FORCE_CATEGORICAL = ["MSSubClass", "MoSold"]

def enforce_data_types(df):
    """
    Belirlenen sütunların tiplerini string'e (kategorik) zorlar
    ve dönüştürülmüş DataFrame'i geri döndürür.
    """
    df = df.copy()
    for col in FORCE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def get_column_types(df):
    """
    Sayısal ve kategorik sütun listelerini döndürür.
    DİKKAT: df bu fonksiyona girmeden önce enforce_data_types'tan geçmiş olmalıdır.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Target column'ı sayısal özellikler arasından çıkar (eğer varsa)
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
        
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    return numeric_cols, categorical_cols

def split_features_target(df, target_col=TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)