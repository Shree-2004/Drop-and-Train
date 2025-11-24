

import streamlit as st
st.set_page_config(page_title="AutoML — Pickle Safe", layout="wide")

import pandas as pd
import numpy as np
import re, ast, json, traceback, tempfile, os
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
from scipy.stats import randint, uniform
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Top-level helper functions (must be at module scope for pickling)
# -------------------------
_LISTLIKE_RE = re.compile(r'^\s*\[.*\]\s*$')

def clean_text(s: object) -> str:
    """Top-level text cleaner."""
    if pd.isna(s):
        return ""
    s = str(s)
    # keep alphanumeric and spaces
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()

def parse_listlike(s: object):
    """Top-level listlike parser (returns list of strings)."""
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if x is not None and str(x).strip()]
    except Exception:
        pass
    if isinstance(s, str) and "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if str(s).strip():
        return [str(s).strip()]
    return []

def flatten_to_1d(X):
    """FunctionTransformer target: flatten (n,1) array-like to 1D list/array"""
    # convert to numpy array then ravel
    return np.asarray(X).ravel()

def detect_task(y):
    """Detect regression vs classification from y series."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return "regression"
    return "classification"

# Small compatibility factory for OneHotEncoder
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# -------------------------
# Preprocessing utilities
# -------------------------
def preprocess_df(df: pd.DataFrame, listlike_threshold: float = 0.2, text_truncate: int = 800) -> pd.DataFrame:
    """
    Create *_text and numeric conversion features.
    Conservative: keep original columns and add engineered ones, finally return engineered set.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                df[col + "_text"] = ""
                continue
            listlike_frac = sample.apply(lambda x: bool(_LISTLIKE_RE.match(str(x)))).mean()
            if listlike_frac > listlike_threshold:
                df[col + "_text"] = df[col].astype(str).apply(lambda v: " ".join([clean_text(x) for x in parse_listlike(v) if x]))
            else:
                df[col + "_text"] = df[col].astype(str).apply(clean_text)

    text_cols = [c for c in df.columns if c.endswith("_text")]
    if text_cols:
        df["all_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
        # truncate to keep TF-IDF safe
        df["all_text"] = df["all_text"].astype(str).str.slice(0, text_truncate)
    return df

def build_preprocessor(df: pd.DataFrame, max_ohe_cardinality: int = 20, max_text_features: int = 300, min_text_docs: int = 5):
    """
    Build ColumnTransformer. Only include text transformer if enough non-empty text documents exist.
    Returns: (preprocessor, text_included_bool)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # text candidate
    text_present = "all_text" in df.columns
    non_empty_text_count = 0
    if text_present:
        non_empty_text_count = int(df["all_text"].astype(str).str.strip().replace("", np.nan).dropna().shape[0])

    cat_cols = [c for c in df.columns if c not in numeric_cols and c != "all_text" and not c.endswith("_text")]
    low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_ohe_cardinality]
    high_card = [c for c in cat_cols if c not in low_card]

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("num", numeric_pipe, numeric_cols))

    if low_card:
        low_cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_onehot())])
        transformers.append(("low_cat", low_cat_pipe, low_card))

    if high_card:
        high_cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
        transformers.append(("high_cat", high_cat_pipe, high_card))

    text_included = False
    if text_present and non_empty_text_count >= min_text_docs:
        # use top-level flatten_to_1d (pickle safe)
        tfidf_pipe = Pipeline([("selector", FunctionTransformer(flatten_to_1d, validate=False)),
                               ("tfidf", TfidfVectorizer(max_features=max_text_features, token_pattern=r"(?u)\b\w+\b"))])
        transformers.append(("text", tfidf_pipe, ["all_text"]))
        text_included = True

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, text_included

# -------------------------
# Safe logistic helper (uses top-level functions only)
# -------------------------
def safe_logistic_fit(preprocessor, X_train, y_train, scoring, cv, n_iter=10):
    """
    Fit Logistic with RandomizedSearchCV on pipeline (preprocessor + model).
    If TF-IDF raises empty vocabulary, attempt fallback without text preprocessor.
    Returns fitted_pipeline (ready-to-predict), cv_score (or None)
    """
    base_model = LogisticRegression(max_iter=5000, random_state=42)
    pipe = Pipeline([("pre", preprocessor), ("model", base_model)])

    param_space = {
        "model__solver": ["saga"],
        "model__penalty": ["l1", "l2"],
        "model__C": uniform(0.01, 5)
    }
    try:
        rs = RandomizedSearchCV(pipe, param_space, n_iter=min(20, max(5, int(n_iter))), cv=int(cv),
                                scoring=scoring, n_jobs=1, random_state=42, error_score="raise")
        rs.fit(X_train, y_train)
        best_est = rs.best_estimator_
        return best_est, (rs.best_score_ if rs.best_score_ is not None else None)
    except Exception as e:
        # detect empty vocabulary error
        msg = str(e).lower()
        if "empty vocabulary" in msg:
            # rebuild preprocessor without the 'text' transformer
            pre = preprocessor
            # create new transformer list removing 'text'
            try:
                new_transformers = [t for t in pre.transformers if t[0] != "text"]
                new_pre = ColumnTransformer(transformers=new_transformers, remainder="drop")
                fallback_pipe = Pipeline([("pre", new_pre), ("model", LogisticRegression(max_iter=3000, random_state=42))])
                fallback_pipe.fit(X_train, y_train)
                return fallback_pipe, None
            except Exception as e2:
                raise e2
        else:
            # fallback: try direct fit with original pipe (no search)
            try:
                pipe.fit(X_train, y_train)
                return pipe, None
            except Exception:
                raise

# -------------------------
# Streamlit UI + main flow
# -------------------------
st.title("AutoML — Pickle-safe, TF-IDF-fallback")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# read
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.write("Preview:")
st.dataframe(df.head())

# choose target
target = st.selectbox("Select target column", df.columns)
if target is None:
    st.stop()

# basic options
n_iter = st.sidebar.slider("Random search iterations (per model)", 5, 100, 20)
cv = st.sidebar.slider("CV folds", 2, 6, 3)
test_size = st.sidebar.slider("Holdout test size", 0.05, 0.4, 0.2, 0.05)
max_ohe_card = st.sidebar.number_input("Max OHE cardinality", min_value=2, max_value=500, value=20, step=1)
max_text_features = st.sidebar.number_input("Max TF-IDF features", min_value=50, max_value=2000, value=300, step=50)
min_text_docs = st.sidebar.number_input("Min non-empty text docs to include TF-IDF", min_value=1, max_value=1000, value=5)

# preprocess
df = df.dropna(subset=[target]).reset_index(drop=True)
df_proc = preprocess_df(df)
# drop any engineered column that is identical to target name
if target in df_proc.columns:
    df_proc.drop(columns=[target], inplace=True, errors="ignore")

# split
X = df_proc
y = df[target].reset_index(drop=True)
task = detect_task(y)
st.write("Detected task:", task)
try:
    strat = y if task == "classification" else None
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    val_frac = 0.2
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_frac, random_state=42, stratify=(y_train_full if task=="classification" else None))
except Exception as e:
    st.error(f"Failed to split dataset: {e}")
    st.stop()

# build preprocessor (pickle-safe flatten function used)
preprocessor, text_included = build_preprocessor(X_train, max_ohe_cardinality=max_ohe_card, max_text_features=max_text_features, min_text_docs=min_text_docs)
st.write("Preprocessor built. Text included in preprocessor:", text_included)

# models dict
models = {}
results = {}

# classification logistic special-case
if task == "classification":
    scoring = "roc_auc" if y.nunique() == 2 else "f1_weighted"
    try:
        log_pipe, log_cv = safe_logistic_fit(preprocessor, X_train, y_train, scoring=scoring, cv=cv, n_iter=n_iter)
        models["logistic"] = log_pipe
        results["logistic"] = {"cv": log_cv}
        st.write("Logistic trained. CV:", log_cv)
    except Exception as e:
        st.warning(f"Logistic training failed: {e}")
        st.exception(e)

# RandomForest
if task == "classification":
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
else:
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42)

rf_pipe = Pipeline([("pre", preprocessor), ("model", rf_model)])
try:
    rf_pipe.fit(X_train, y_train)
    models["rf"] = rf_pipe
    results["rf"] = {}
    st.write("RandomForest trained.")
except Exception as e:
    st.error(f"RandomForest failed: {e}")
    st.exception(e)

# HGB
try:
    if task == "classification":
        hgb_model = HistGradientBoostingClassifier(random_state=42)
    else:
        hgb_model = HistGradientBoostingRegressor(random_state=42)
    hgb_pipe = Pipeline([("pre", preprocessor), ("model", hgb_model)])
    hgb_pipe.fit(X_train, y_train)
    models["hgb"] = hgb_pipe
    results["hgb"] = {}
    st.write("HGB trained.")
except Exception as e:
    st.warning(f"HGB training failed: {e}")
    st.exception(e)

if not models:
    st.error("No models trained successfully.")
    st.stop()

# Evaluate on holdout
metrics = {}
for name, pipe in models.items():
    try:
        preds = pipe.predict(X_holdout)
        if task == "classification":
            if y.nunique() == 2 and hasattr(pipe, "predict_proba"):
                score = roc_auc_score(y_holdout, pipe.predict_proba(X_holdout)[:,1])
            else:
                score = f1_score(y_holdout, preds, average="weighted")
        else:
            score = np.sqrt(mean_squared_error(y_holdout, preds))
        metrics[name] = float(score)
        st.write(f"{name} holdout score: {score:.4f}")
    except Exception as e:
        st.warning(f"Evaluation failed for {name}: {e}")
        st.exception(e)

# pick best
if task == "classification":
    best_name = max(metrics, key=metrics.get)
else:
    best_name = min(metrics, key=metrics.get)

st.success(f"Best model: {best_name}")

# Save best model to file (pickle-safe because we only used top-level funcs)
out_dir = tempfile.mkdtemp(prefix="automl_")
fname = os.path.join(out_dir, f"best_model_{best_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
try:
    joblib.dump(models[best_name], fname)
    with open(fname, "rb") as f:
        bytes_data = f.read()
    st.download_button("Download best model (.joblib)", data=bytes_data, file_name=os.path.basename(fname))
    st.write("Model saved to:", fname)
except Exception as e:
    st.error("Failed to save model:")
    st.exception(e)
