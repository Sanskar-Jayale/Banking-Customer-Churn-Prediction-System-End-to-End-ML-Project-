# src/pipeline/train_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training():

    ingestion = DataIngestion(
    "D:/Pro/End-to-end-churn-model/data/Bank-Customer-Attrition-Insights-Data.csv"
)
    df = ingestion.load_data()

    X = df.drop("exited", axis=1)
    y = df["exited"]

    transformer = DataTransformation()
    X_transformed = transformer.fit_transform(X, version="v1")

    trainer = ModelTrainer()
    trainer.train(X_transformed, y, version="v1")

if __name__ == "__main__":
    run_training()
