import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features

# Paths & Constants
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
PIPELINE_LUXURY_PATH = MODEL_DIR / "pipeline_luxury.joblib"
LUXURY_CLASSIFIER_PATH = MODEL_DIR / "luxury_classifier.joblib"
LUXURY_THRESHOLD_PATH = MODEL_DIR / "luxury_threshold.json"
PIPELINE_LOWER_PATH = MODEL_DIR / "pipeline_lower.joblib"
PIPELINE_UPPER_PATH = MODEL_DIR / "pipeline_upper.joblib"
TARGET = "price"

# Chosen on the old validation split (scripts/evaluate_stratified.py,
# 2026-10-11) and checked, not re-tuned, on the new one: it holds 19 luxury
# rows, too few to tune on. The classifier's probabilities are inflated by
# scale_pos_weight (~98), so at 0.7 most routed properties weren't luxury and
# jumped to a luxury-model price.
DEFAULT_ROUTING_THRESHOLD = 0.9

# Four disjoint splits, drawn once with SPLIT_SEED and shared by every script:
#   train        70%  fits every model and the preprocessor
#   validation   10%  early stopping, the Optuna search, the routing threshold
#   calibration   5%  the conformal margin of the price range, nothing else
#   test         15%  sealed: only the final figures read it (monitoring/metrics.py,
#                     scripts/evaluate_stratified.py --test, the range coverage)
# Until 2026-11-10 there was one 80/20 split (LEGACY_VALIDATION_SIZE) and its
# 20% did everything validation does now, plus the published figures. The test
# rows are drawn from that old 80% only, so no earlier choice (the 0.9
# threshold, the October search, the blend comparison) was ever scored on them.
# Drawn with the same seed, a fresh 15% would sit entirely inside the old 20%.
TEST_SIZE = 0.15
CALIBRATION_SIZE = 0.05
VALIDATION_SIZE = 0.10
LEGACY_VALIDATION_SIZE = 0.2
SPLIT_SEED = 42

QUANTILE_LOWER = 0.10
QUANTILE_UPPER = 0.90
# Coverage the 10th-90th percentile range is calibrated to.
INTERVAL_COVERAGE = QUANTILE_UPPER - QUANTILE_LOWER
INTERVAL_CALIBRATION_PATH = MODEL_DIR / "interval_calibration.json"


def load_dataset(path=DATA_PATH):
    """The cleaned listings with every engineered feature, as training sees them."""
    return add_features(pd.read_json(path))


def split_all(frame):
    """
    (train, validation, calibration, test), the four splits described at the
    top of this file. Each one is a slice of `frame`, so its index points back
    to the original rows.
    """
    n = len(frame)
    legacy_train, _ = train_test_split(frame, test_size=LEGACY_VALIDATION_SIZE,
                                       random_state=SPLIT_SEED)
    _, test = train_test_split(legacy_train, test_size=round(n * TEST_SIZE),
                               random_state=SPLIT_SEED)
    rest = frame.drop(index=test.index)
    rest, calibration = train_test_split(rest, test_size=round(n * CALIBRATION_SIZE),
                                         random_state=SPLIT_SEED)
    train, validation = train_test_split(rest, test_size=round(n * VALIDATION_SIZE),
                                         random_state=SPLIT_SEED)
    return train, validation, calibration, test


def split(frame):
    """
    (train, validation): the rows the models are fitted on and the rows they
    early-stop and get tuned on. Scripts that need the training vocabulary or
    the drift baseline use this; final figures use split_all()'s test split.
    """
    train, validation, _, _ = split_all(frame)
    return train, validation


def price_bounds(train_prices):
    """
    (1st, 99th) percentile of the TRAINING prices. They set the clip of the
    target and the luxury price; taken on all rows they would let the
    validation prices shape what the models are scored against.
    """
    return float(train_prices.quantile(0.01)), float(train_prices.quantile(0.99))


def clipped_log_target(prices, bounds):
    """
    log1p of the price clipped to `bounds` (price_bounds of the training rows):
    the target of the standard and quantile models. The upper bound is the
    luxury price, so those models cannot predict above it, which is why the
    luxury path exists.
    """
    low, high = bounds
    return np.log1p(prices.clip(low, high))


def price_tiers(luxury_price):
    """Below €1M, €1M up to the luxury price, and the luxury segment."""
    return [("< €1M", 0, 1_000_000),
            (f"€1M - €{luxury_price / 1e6:.1f}M", 1_000_000, luxury_price),
            (f">= €{luxury_price / 1e6:.1f}M", luxury_price, float("inf"))]


def conformal_margin(lower_log, upper_log, y_log, coverage):
    """
    Conformalized quantile regression: the log-scale amount to widen (or, if
    negative, narrow) both bounds so that `coverage` of the calibration rows
    fall inside. Raw quantile models covered only ~71% instead of 80%; the
    finite-sample quantile level gives the coverage guarantee on new rows, as
    long as the calibration rows played no part in fitting the bounds (no early
    stopping on them either).
    """
    scores = np.maximum(lower_log - y_log, y_log - upper_log)
    level = min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def quantile_regressor(alpha):
    """
    One bound of the price range: an XGBoost model of the `alpha` quantile of
    the log-price. Both bounds share this configuration, so they can only
    differ by the quantile they estimate.
    """
    return XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=42,
        early_stopping_rounds=50
    )


def build_preprocessor():
    """
    Constructs and returns a sklearn ColumnTransformer pipeline.
    Handles imputation and encoding for both numerical and categorical features.
    """
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", TargetEncoder())])
    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", cat, CATEGORICAL_FEATURES)
    ])


def train():
    """
    Executes the full training pipeline:
    1. Loads and engineers features.
    2. Trains standard regression and quantile interval models.
    3. Trains a luxury classification model and an optional luxury-specific regressor.
    4. Persists all models and configuration thresholds to disk.
    """
    df = load_dataset()
    X = df.drop(columns=[TARGET])
    X_train, X_val, X_calib, X_test = split_all(X)
    bounds = price_bounds(df.loc[X_train.index, TARGET])
    y = clipped_log_target(df[TARGET], bounds)
    y_train, y_val = y.loc[X_train.index], y.loc[X_val.index]
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train, y_train)
    X_val_p = preprocessor.transform(X_val)

    # Standard regressor, with the parameters src/optimize.py found on the
    # train and validation splits (2026-11-10, seeded search, 50 trials).
    model = XGBRegressor(
        n_estimators=714,
        learning_rate=0.08075316093557,
        max_depth=8,
        reg_lambda=6.337637256217289,
        subsample=0.8078618752326912,
        colsample_bytree=0.517705564570911,
        objective="reg:absoluteerror",
        random_state=42,
        early_stopping_rounds=50
    )
    model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", model)]), PIPELINE_PATH)

    # Prediction Interval (Quantile Regression) 
    lower_model = quantile_regressor(QUANTILE_LOWER)
    lower_model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)
    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", lower_model)]), PIPELINE_LOWER_PATH)

    upper_model = quantile_regressor(QUANTILE_UPPER)
    upper_model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)
    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", upper_model)]), PIPELINE_UPPER_PATH)

    # Luxury Routing Classifier 
    luxury_price_threshold = bounds[1]
    y_lux = (df[TARGET] > luxury_price_threshold).astype(int)

    y_lux_train = y_lux.loc[X_train.index]
    pos_weight = (y_lux_train == 0).sum() / max((y_lux_train == 1).sum(), 1)

    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=4,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42
    )

    clf.fit(X_train_p, y_lux.loc[X_train.index])

    # Train the luxury-segment regressor on luxury-labeled rows only
    X_lux_train = X_train[y_lux.loc[X_train.index] == 1]
    # Unclipped prices: y_train is clipped at the 99th percentile, which is the
    # luxury threshold itself, so every luxury row would share the same target
    # and the regressor would learn a constant.
    y_lux_train_price = np.log1p(df.loc[X_lux_train.index, TARGET])
    if len(X_lux_train) >= 20:
        X_lux_train_p = preprocessor.transform(X_lux_train)
        luxury_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:absoluteerror",
            random_state=42
        )
        luxury_model.fit(X_lux_train_p, y_lux_train_price)
        joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", luxury_model)]), PIPELINE_LUXURY_PATH)
        luxury_model_available = True
    else:
        print(f"Warning: only {len(X_lux_train)} luxury rows in the training split — "
              f"skipping luxury regressor training (need >= 20).")
        luxury_model_available = False

    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", clf)]), LUXURY_CLASSIFIER_PATH)

    with open(LUXURY_THRESHOLD_PATH, "w") as f:
        json.dump({
            "luxury_price_threshold": float(luxury_price_threshold),
            "routing_probability_threshold": DEFAULT_ROUTING_THRESHOLD,
            "luxury_model_available": luxury_model_available
        }, f)

    # Calibrate the range on the calibration split and measure it on the test
    # split, keeping only rows the API would give a range to (routed to the
    # standard model), against real, unclipped prices. Neither split served
    # early stopping, so the margin and the coverage are both out of sample.
    def standard_rows(X_part):
        X_part_p = preprocessor.transform(X_part)
        keep = clf.predict_proba(X_part_p)[:, 1] < DEFAULT_ROUTING_THRESHOLD
        return (lower_model.predict(X_part_p)[keep], upper_model.predict(X_part_p)[keep],
                np.log1p(df.loc[X_part.index, TARGET]).to_numpy()[keep])

    calib_lower, calib_upper, calib_y = standard_rows(X_calib)
    test_lower, test_upper, test_y = standard_rows(X_test)
    margin = conformal_margin(calib_lower, calib_upper, calib_y, INTERVAL_COVERAGE)

    def coverage(m):
        return float(np.mean((test_y >= test_lower - m) & (test_y <= test_upper + m)))

    with open(INTERVAL_CALIBRATION_PATH, "w") as f:
        json.dump({
            "target_coverage": round(INTERVAL_COVERAGE, 2),
            "log_margin": margin,
            "calibration_rows": len(calib_y),
            "check_rows": len(test_y),
            "check_coverage_raw": round(coverage(0.0), 4),
            "check_coverage_calibrated": round(coverage(margin), 4),
        }, f, indent=1)
    print(f"Price range: coverage {coverage(0.0):.1%} -> {coverage(margin):.1%} "
          f"on {len(test_y)} test rows (log margin {margin:+.3f})")

    print("Training complete. Models updated.")


if __name__ == "__main__":
    """
    Entry point for the training script.
    """
    train()