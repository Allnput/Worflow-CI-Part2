import argparse
import joblib
import numpy as np
import pandas as pd
import os
import mlflow
import mlflow.sklearn

# Gunakan path absolute untuk MLflow tracking
mlruns_path = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "MLProjects", "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.set_experiment("Maintenance-Prediction-CI")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def main(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Target", "Failure Type"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params_list = [
        {"C": 0.01, "solver": "lbfgs"},
        {"C": 0.1, "solver": "lbfgs"},
        {"C": 1.0, "solver": "lbfgs"},
    ]

    best_f1 = 0
    best_model = None
    best_params = None
    best_y_pred = None

    for params in params_list:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params
            best_y_pred = y_pred

    # Metrics
    acc = accuracy_score(y_test, best_y_pred)
    recall = recall_score(y_test, best_y_pred)
    precision = precision_score(y_test, best_y_pred)
    f1 = f1_score(y_test, best_y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, best_y_pred).ravel()

    # Log ke MLflow
    os.environ.pop("MLFLOW_RUN_ID", None)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("true_negative", tn)
        mlflow.log_metric("false_positive", fp)
        mlflow.log_metric("false_negative", fn)
        mlflow.log_metric("true_positive", tp)

        mlflow.sklearn.log_model(
            best_model, "model",
            input_example=X_test.iloc[:5]
        )
        
    print("Best Parameters:", best_params)
    print(f"Accuracy  : {acc}")
    print(f"Recall    : {recall}")
    print(f"Precision : {precision}")
    print(f"F1-score  : {f1}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, best_y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
