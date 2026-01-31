Banking Customer Churn Prediction – End-to-End ML Project
📌 Overview

This project implements an end-to-end machine learning pipeline to predict customer churn in the banking domain using customer demographic and account-level data.
The solution follows production-grade ML practices, including modular pipelines, model versioning, experiment tracking, and REST API deployment.

🎯 Problem Statement

Customer churn is a critical challenge for banks, directly impacting revenue and customer lifetime value.
This project aims to:

Predict whether a customer is likely to churn

Enable data-driven customer retention and risk mitigation

🛠 Tech Stack

Programming Language: Python

Machine Learning: Scikit-learn

Experiment Tracking: MLflow

API Framework: FastAPI

Containerization: Docker

Model Serialization: Joblib / Pickle

🧱 Project Architecture
End-to-end-churn-model/
│
├── app.py                     # FastAPI application
├── requirements.txt
├── Dockerfile
│
├── artifacts/                 # Saved models & preprocessors
│   └── v1/
│       ├── model.pkl
│       └── preprocessor.pkl
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│
└── README.md

🔄 ML Pipeline

Data Ingestion

Load banking customer dataset

Handle schema and basic validation

Data Transformation

Feature encoding & scaling

Save reusable preprocessing pipeline

Model Training

Train classification models

Hyperparameter tuning

Evaluate accuracy and stability

Track experiments using MLflow

Model Versioning

Store models under versioned artifacts (v1, v2, etc.)

📊 Model Performance

Classification models trained with hyperparameter tuning

Achieved strong predictive accuracy and stable performance

Experiments tracked and reproducible via MLflow

🚀 API Usage (FastAPI)
Root Endpoint
GET /


Provides API information and available routes.

Prediction Endpoint
POST /predict


Sample Request:

{
  "creditscore": 650,
  "geography": "France",
  "gender": "Male",
  "age": 40,
  "balance": 60000,
  "cardtype": "Gold",
  "estimatedsalary": 50000
}


Sample Response:

{
  "churn_prediction": 1
}

📖 API Documentation

Once the server is running:

Swagger UI: http://localhost:8000/docs

ReDoc UI: http://localhost:8000/redoc

🐳 Docker & Deployment
Build Docker Image
docker build -t churn-prediction-api .

Run Container
docker run -p 7860:7860 churn-prediction-api

Hugging Face Spaces

Deployed using Docker-based FastAPI service

Compatible with Hugging Face Spaces runtime

🏦 Banking Relevance

Designed specifically for banking customer churn analysis

Supports customer retention strategies

Aligns with risk mitigation and data-driven decision-making

Built following SDLC and production engineering best practices

📌 Future Enhancements

Add transaction-level data

Model monitoring & drift detection

CI/CD integration

Authentication & authorization for API

👤 Author

Sanskar Jayale
Aspiring Data Scientist / ML Engineer
Focused on Banking, Risk Analytics, and Production ML Systems