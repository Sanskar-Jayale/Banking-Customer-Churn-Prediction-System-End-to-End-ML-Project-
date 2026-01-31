from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a bank customer will churn",
    version="1.0.0"
)

class ChurnRequest(BaseModel):
    creditscore: int
    geography: str
    gender: str
    age: int
    balance: float
    cardtype: str
    estimatedsalary: float


@app.get("/")
def root():
    return {
        "message": "Welcome to the Churn Prediction API",
        "description": "This API predicts whether a bank customer is likely to churn.",
        "endpoints": {
            "/docs": "Interactive Swagger UI to test the API (recommended for developers)",
            "/redoc": "Readable API documentation (good for overview)",
            "/predict": {
                "method": "POST",
                "description": "Send customer details and receive churn prediction"
            }
        },
        "status": "API is up and running"
    }



@app.post("/predict")
def predict_churn(data: ChurnRequest):
    pipeline = PredictPipeline()
    result = pipeline.predict(data.dict())
    return {"churn_prediction": result}
