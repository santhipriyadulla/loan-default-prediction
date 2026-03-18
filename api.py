from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle

# LOAD FILES
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# CREATE APP
app = FastAPI()

# ✅ ADD CORS HERE (CORRECT PLACE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HOME
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running 🚀"}

# PREDICT
@app.post("/predict")
def predict(data: dict):
    try:
        input_data = data.copy()

        for col, le in le_dict.items():
            if col in input_data:
                input_data[col] = le.transform([input_data[col]])[0]

        features = [input_data[col] for col in feature_names]
        features_array = np.array([features])

        scaled_data = scaler.transform(features_array)
        prediction = model.predict(scaled_data)[0]

        return {
            "prediction": int(prediction),
            "result": "Default" if prediction == 1 else "No Default"
        }

    except Exception as e:
        return {"error": str(e)}
