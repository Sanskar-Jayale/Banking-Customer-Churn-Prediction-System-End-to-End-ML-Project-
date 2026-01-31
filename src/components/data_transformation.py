# src/components/data_transformation.py
import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:

    def _validate_columns(self, X, columns):
        missing = set(columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def get_preprocessor(self, X):

        numerical_features = [
            "creditscore",
            "age",
            "balance",
            "estimatedsalary"
        ]

        categorical_features = [
            "geography",
            "gender",
            "cardtype"
        ]

        self._validate_columns(X, numerical_features + categorical_features)

        num_pipeline = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        cat_pipeline = Pipeline(
            steps=[("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ]
        )

    def fit_transform(self, X, version="v1"):

        preprocessor = self.get_preprocessor(X)
        X_transformed = preprocessor.fit_transform(X)

        # ✅ CREATE DIRECTORIES FIRST
        os.makedirs(f"artifacts/{version}", exist_ok=True)
        os.makedirs("artifacts/latest", exist_ok=True)

        with open(f"artifacts/{version}/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        with open("artifacts/latest/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        return X_transformed
