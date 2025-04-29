import torch
import numpy as np
from clearml import Task
from fastapi import FastAPI
from model import FraudDetectionModel


app = FastAPI()
print("Loading model...")
model = FraudDetectionModel()
model.load_state_dict(torch.load("fraud_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.get("/")
def home():
    return "Welcome to Fraud Detection"

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    features = torch.tensor(features, dtype = torch.float32)

    prediction = model(features).item()
    
    return {"fraud_probability": prediction}