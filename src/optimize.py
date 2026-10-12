"""
Optuna hyperparameter search for the standard regressor.

Scores each trial on exactly the setup src/train.py ships: the same rows, the
same split, the same clipped log target, the same preprocessor and the same
early stopping on the validation split. A search on any other setup tunes a
model that is never trained. The best parameters are copied into train.py by
hand, so a search never changes the shipped model on its own.

Run (optuna is not a runtime dependency):
    pip install optuna
    python src/optimize.py
"""
import numpy as np
from xgboost import XGBRegressor

from train import TARGET, build_preprocessor, clipped_log_target, load_dataset, price_bounds, split


def prepare():
    """Preprocessed training and validation arrays, as train.py builds them."""
    df = load_dataset()
    X = df.drop(columns=[TARGET])
    X_train, X_val = split(X)
    y = clipped_log_target(df[TARGET], price_bounds(df.loc[X_train.index, TARGET]))
    y_train, y_val = y.loc[X_train.index], y.loc[X_val.index]
    preprocessor = build_preprocessor()
    return (preprocessor.fit_transform(X_train, y_train), y_train,
            preprocessor.transform(X_val), y_val)


def objective(trial, data):
    """Validation MAE on the log price for one set of hyperparameters."""
    X_train_p, y_train, X_val_p, y_val = data
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 500, 1500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0),
        objective="reg:absoluteerror",
        random_state=42,
        early_stopping_rounds=50,
    )
    model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)
    return float(np.mean(np.abs(model.predict(X_val_p) - y_val)))


if __name__ == "__main__":
    import optuna

    data = prepare()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, data), n_trials=50)

    print("\n--- Optimization finished ---")
    print(f"Best parameters: {study.best_params}")
