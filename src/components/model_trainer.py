# src/components/model_trainer.py
import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class ModelTrainer:

    def train(self, X, y, version="v1"):

        # train-test split (IMPORTANT for MLflow)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # hyperparameters
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42
        }

        with mlflow.start_run(run_name=f"rf_churn_{version}"):

            # log params
            mlflow.log_params(params)

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # predictions
            y_pred = model.predict(X_test)

            # metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # save locally (versioned)
            os.makedirs(f"artifacts/{version}", exist_ok=True)
            os.makedirs("artifacts/latest", exist_ok=True)

            with open(f"artifacts/{version}/model.pkl", "wb") as f:
                pickle.dump(model, f)

            with open("artifacts/latest/model.pkl", "wb") as f:
                pickle.dump(model, f)

            # log model to MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="ChurnPredictionModel"
            )

        return model
