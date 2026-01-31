# test_predict.py
from src.pipeline.predict_pipeline import PredictPipeline

sample_input = {
    "creditscore": 619,
    "geography": "France",
    "gender": "Female",
    "age": 42,
    "balance": 0.0,
    "cardtype": "Platinum",
    "estimatedsalary": 101348.88
}

pipeline = PredictPipeline()
prediction = pipeline.predict(sample_input)

print("Prediction:", prediction)
