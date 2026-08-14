import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

NONE_FILL_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

def fill_none_categoricals(df):
    df = df.copy()
    for col in NONE_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    return df

ZERO_FILL_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
]

def fill_zero_numerics(df):
    df = df.copy()
    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


class NumericImputer:
    def __init__(self, columns):
        self.columns = columns
        self.fill_values = {}

    def fit(self, df):
        for col in self.columns:
            if col in df.columns:
                self.fill_values[col] = df[col].median()
        return self

    def transform(self, df):
        df = df.copy()
        for col, value in self.fill_values.items():
            df[col] = df[col].fillna(value)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

class CategoricalImputer:
    def __init__(self, columns):
        self.columns = columns
        self.fill_values = {}

    def fit(self, df):
        for col in self.columns:
            if col in df.columns:
                self.fill_values[col] = df[col].mode()[0]
        return self

    def transform(self, df):
        df = df.copy()
        for col, value in self.fill_values.items():
            df[col] = df[col].fillna(value)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

MEDIAN_FILL_COLS = ["LotFrontage"]
MODE_FILL_COLS = [
    "Electrical", "KitchenQual",
    "MSZoning", "Utilities", "Exterior1st", "Exterior2nd",
    "Functional", "SaleType",
]


class DataCleaner:
    def __init__(self):
        self.num_imputer = NumericImputer(columns=MEDIAN_FILL_COLS)
        self.cat_imputer = CategoricalImputer(columns=MODE_FILL_COLS)

    def fit_transform(self, df):
        df = fill_none_categoricals(df)
        df = fill_zero_numerics(df)
        df = self.num_imputer.fit_transform(df)
        df = self.cat_imputer.fit_transform(df)
        return df

    def transform(self, df):
        df = fill_none_categoricals(df)
        df = fill_zero_numerics(df)
        df = self.num_imputer.transform(df)
        df = self.cat_imputer.transform(df)
        return df

def remove_outliers(df, target_col="SalePrice"):
    df = df.copy()
    mask = (df["GrLivArea"] > 4000) & (df[target_col] < 300000)
    return df[~mask]


QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPPINGS = {
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "PoolQC": QUALITY_MAP,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LandSlope": {"Gtl": 1, "Mod": 2, "Sev": 3},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive": {"N": 1, "P": 2, "Y": 3},
}

def encode_ordinal(df):
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


ONEHOT_COLS = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "Utilities",
    "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType",
    "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical",
    "Functional", "GarageType", "Fence", "MiscFeature", "SaleType",
    "SaleCondition", "MoSold",
]


class OneHotEncoderWrapper:

    def __init__(self, columns=ONEHOT_COLS):
        self.columns = columns
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, df):
        columns = [c for c in self.columns if c in df.columns]
        self.columns = columns
        self.encoder.fit(df[columns])
        return self

    def transform(self, df):
        df = df.copy()
        encoded = self.encoder.transform(df[self.columns])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(self.columns),
            index=df.index,
        )
        df = df.drop(columns=self.columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


class ScalerWrapper:

    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None

    def fit(self, df):
        self.columns = [
            c for c in df.select_dtypes(include=["int64", "float64"]).columns
            if not set(df[c].unique()).issubset({0, 1})
        ]
        self.scaler.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

class FullPreprocessor:

    def __init__(self):
        self.cleaner = DataCleaner()
        self.ohe = OneHotEncoderWrapper()
        self.scaler = ScalerWrapper()

    def fit_transform(self, df):
        df = self.cleaner.fit_transform(df)
        df = encode_ordinal(df)
        df = self.ohe.fit_transform(df)
        df = self.scaler.fit_transform(df)
        return df

    def transform(self, df):
        df = self.cleaner.transform(df)
        df = encode_ordinal(df)
        df = self.ohe.transform(df)
        df = self.scaler.transform(df)
        return df

DROP_MULTICOLLINEAR_COLS = ["GarageArea", "1stFlrSF"]


def drop_multicollinear(df):
    df = df.copy()
    cols_to_drop = [c for c in DROP_MULTICOLLINEAR_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)

FEATURE_IMPORTANCE_THRESHOLD = 0.001


class FeatureSelector:

    def __init__(self, threshold=FEATURE_IMPORTANCE_THRESHOLD):
        self.threshold = threshold
        self.selected_columns = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        self.selected_columns = importances[importances >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X[self.selected_columns]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)