from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).with_name("mushroom_v44_train.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "class"

RAW_NUMERIC = ["0", "1", "2"]
RATIOS = ["height-width", "cap-height", "cap-width"]
NUM_REPAIRED = RAW_NUMERIC + RATIOS

CATEGORICAL_ALL = ["cap-surface", "gill-spacing", "stem-root", "stem-surface", "season"]
CATEGORICAL_LR = ["cap-surface", "gill-spacing", "stem-root", "stem-surface"]
CATEGORICAL_ET = ["cap-surface", "gill-spacing", "stem-surface"]
CATEGORICAL_HGB = CATEGORICAL_ALL

DECODE = {
    "cap-surface": {
        "s": "smooth",
        "t": "sticky",
        "w": "wrinkled",
        "e": "fleshy",
        "i": "fibrous",
        "g": "grooves",
        "k": "silky",
        "l": "leathery",
        "y": "scaly",
        "h": "shiny",
        "d": "dry?",
    },
    "gill-spacing": {"c": "close", "d": "distant", "f": "none"},
    "stem-root": {
        "s": "swollen",
        "b": "bulbous",
        "r": "rooted",
        "c": "club",
        "f": "none?",
    },
    "stem-surface": {
        "i": "fibrous",
        "y": "scaly",
        "f": "none",
        "s": "smooth",
        "h": "shiny",
        "g": "grooves",
        "k": "silky",
        "t": "sticky",
    },
    "season": {"w": "winter", "s": "spring", "u": "summer", "a": "autumn"},
}


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def load_split(path=DATA_PATH, stratify=False):
    df = pd.read_csv(path)
    y = df.pop(TARGET)
    return train_test_split(
        df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if stratify else None,
    )


def decode(X):
    X = X.copy()
    for col, mapping in DECODE.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(X[col])
    return X


class MushroomRepair(BaseEstimator, TransformerMixin):
    def __init__(self, cap_q=0.99, add_ratios=True, restore_width=True):
        self.cap_q = cap_q
        self.add_ratios = add_ratios
        self.restore_width = restore_width

    def fit(self, X, y=None):
        missing = [c for c in RAW_NUMERIC if c not in X.columns]
        if missing:
            raise ValueError(f"MushroomRepair expects {RAW_NUMERIC}; missing {missing}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.cap_upper_ = X["0"].quantile(self.cap_q)
        self.stem_min_ = X["1"].min()
        return self

    def transform(self, X):
        X = X.copy()
        cap = np.log1p(X["0"].clip(lower=0, upper=self.cap_upper_))
        height = (X["1"] - self.stem_min_).clip(lower=0)
        width = np.exp(-X["2"])

        X["0"] = cap
        X["1"] = height
        X["2"] = width if self.restore_width else X["2"]

        if self.add_ratios:
            safe_height = height.where(height > 0)
            X["height-width"] = height / width
            X["cap-height"] = cap / safe_height
            X["cap-width"] = cap / width
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(
            self.feature_names_in_ if input_features is None else input_features
        )
        return np.asarray(base + (RATIOS if self.add_ratios else []), dtype=object)
