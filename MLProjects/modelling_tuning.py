# -- coding: utf-8 --
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def main():
    df = pd.read_csv("Predictive_Maintenance_Preproces.csv")

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

    for params in params_list:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            **params
        )
        model.fit(X_train, y_train)
        f1 = f1_score(y_test, model.predict(X_test))

        mlflow.log_metric(f"f1_C_{params['C']}", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params

    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_f1)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

if __name__ == "__main__":
    main()
