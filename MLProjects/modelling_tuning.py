# -- coding: utf-8 --
import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import mlflow
import mlflow.sklearn

if __name__ == "_main_":
    with mlflow.start_run():
        df = pd.read_csv("Predictive_Maintenance_Preproces.csv")

        X = df.drop(columns=['Target', 'Failure Type'])
        y = df['Target']

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
        best_y_pred = None
        best_params = None

        for params in params_list:

            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                **params
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print("\n=== Training Result ===")
            print(f"Params      : {params}")
            print(f"Accuracy    : {acc:.4f}")
            print(f"Precision   : {prec:.4f}")
            print(f"Recall      : {rec:.4f}")
            print(f"F1-score    : {f1:.4f}")

            mlflow.log_metric(f"accuracy_C_{params['C']}", acc)
            mlflow.log_metric(f"precision_C_{params['C']}", prec)
            mlflow.log_metric(f"recall_C_{params['C']}", rec)
            mlflow.log_metric(f"f1_score_C_{params['C']}", f1)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_y_pred = y_pred
                best_params = params

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)

        cm = confusion_matrix(y_test, best_y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model"
        )

    print("\n=== MLflow logging completed ===")
