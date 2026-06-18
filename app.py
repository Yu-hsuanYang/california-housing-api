from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

# Define the same model architecture used during training
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

# Load trained model
model = HousingRegressionModel()

model.load_state_dict(
    torch.load(
        "california_housing_regression_model.pth",
        map_location=torch.device("cpu")
    )
)

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
        prediction = model(features_tensor)

    return jsonify({
        "predicted_house_value": float(prediction.item())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
