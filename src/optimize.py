import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from train import build_preprocessor
from features import add_features

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    Loads data, performs feature engineering, trains an XGBoost model
    within a pipeline, and returns the mean absolute error for minimization.
    """
    # Load and prepare the data
    df = pd.read_json("data/clean/cleaned_data.json")
    df = add_features(df)
    
    # Drop columns not necessary for training (IDs, addresses, free text)
    cols_to_drop = ["price", "property_id", "address", "nearest_city"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = np.log1p(df["price"])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use the preprocessor defined in train.py
    preprocessor = build_preprocessor()
    
    # Parameters to optimize
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
    }

    # Complete pipeline: preprocessing + model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(**param, objective="reg:absoluteerror", random_state=42))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    # Return the Mean Absolute Error (MAE) as the metric to minimize
    return np.mean(np.abs(preds - y_val))

if __name__ == "__main__":
    """
    Main execution block to start the Optuna optimization process.
    Runs the study for a defined number of trials and prints the best results.
    """
    # Launch optimization with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("\n--- Optimization finished ---")
    print(f"Best parameters: {study.best_params}")