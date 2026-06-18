from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib

class HousingRegressionModel(nn.Module):
    def __init__(self):
        super(HousingRegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

app = Flask(__name__)

model = joblib.load("california_housing_regression_model.pkl")
model.eval()

@app.route("/")
def home():
    return "California Housing Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        data["MedInc"],
        data["HouseAge"],
        data["AveRooms"],
        data["AveBedrms"],
        data["Population"],
        data["AveOccup"],
        data["Latitude"],
        data["Longitude"]
    ]], dtype=np.float32)

    features_tensor = torch.tensor(features)

    with torch.no_grad():
        prediction = model(features_tensor).item()

    return jsonify({
        "predicted_house_value": prediction
    })

if __name__ == "__main__":
    app.run()
