from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

class ChurnRequest(BaseModel):
    creditscore: int
    geography: str
    gender: str
    age: int
    balance: float
    cardtype: str
    estimatedsalary: float

@app.post("/predict")
def predict_churn(data: ChurnRequest):
    pipeline = PredictPipeline()
    result = pipeline.predict(data.dict())
    return {"churn_prediction": result}
