import json

import pandas as pd
from joblib import dump
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_evaluate_save(
    train_data_path="data/train.csv",
    test_data_path="data/test.csv",
    target_name="price",
    model_path="models/model.joblib",
    metrics_path="metrics.json",
):
    """
    Loads train/test data, trains SGC regressor, evaluates on test set,
    saves model and metrics (RMSE) as JSON.
    Args:
        train_data_path: Path to training data CSV (default: "train.csv").
        test_data_path: Path to testing data CSV (default: "test.csv").
        target_name: Name of the target column (default: "price").
        model_path: Path to save the trained model (default: "model/model.joblib").
        metrics_path: Path to save the evaluation metrics (default: "metrics.json").
    """

    # Load data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Separate features and target
    X_train, y_train = train_data.drop(target_name, axis=1), train_data[target_name]
    X_test, y_test = test_data.drop(target_name, axis=1), test_data[target_name]

    # Create SGC regressor model with preprocessing pipeline
    model = SGDRegressor(loss="squared_error")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Save the model
    dump(model, model_path)

    # Save metrics as JSON
    metrics = {"rmse": rmse}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model trained and saved to: {model_path}")
    print(f"Test set RMSE: {rmse:.4f}")
    print(f"Metrics saved to: {metrics_path}")


# Set file paths and run training/evaluation/saving
train_evaluate_save()
print("This is a new message")
