import numpy as np
import pandas as pd
import re
import unicodedata
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def _log_amount(X):
    return np.log1p(np.abs(X.astype(float))).values.reshape(-1, 1)


def _inout_binary(X):
    return (X.str.strip().str.upper().isin({"IN", "C", "CR", "CREDIT", "ENTRADA"})).astype(int).values.reshape(-1, 1)


def _currency_binary(X):
    return (X.str.strip().str.upper() == "USD").astype(int).values.reshape(-1, 1)


def _day_of_month(X):
    return pd.to_datetime(X, dayfirst=True, errors="coerce").dt.day.values.reshape(-1, 1)


def _normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "tfidf_word",
                TfidfVectorizer(
                    preprocessor=_normalize_text,
                    ngram_range=(1, 2),
                    max_features=7000,
                    sublinear_tf=True,
                ),
                "Description",
            ),
            (
                "tfidf_char",
                TfidfVectorizer(
                    preprocessor=_normalize_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=3000,
                    sublinear_tf=True,
                ),
                "Description",
            ),
            ("amount", Pipeline([
                ("log", FunctionTransformer(_log_amount, validate=False)),
                ("scale", StandardScaler()),
            ]), "Amount"),
            ("inout", FunctionTransformer(_inout_binary, validate=False), "InOut"),
            ("dom", FunctionTransformer(_day_of_month, validate=False), "Date"),
            ("currency", FunctionTransformer(_currency_binary, validate=False), "Currency"),
            ("source", OneHotEncoder(handle_unknown="ignore"), ["Source"]),
        ],
        remainder="drop",
    )
