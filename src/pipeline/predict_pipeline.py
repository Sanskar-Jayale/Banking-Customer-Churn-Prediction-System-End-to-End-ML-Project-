# src/pipeline/predict_pipeline.py
import pickle
import pandas as pd

class PredictPipeline:

    def predict(self, data: dict):

        df = pd.DataFrame([data])

        with open("artifacts/latest/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        with open("artifacts/latest/model.pkl", "rb") as f:
            model = pickle.load(f)

        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)

        return int(prediction[0])
