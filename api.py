# ================================
# IMPORT LIBRARIES
# ================================
from fastapi import FastAPI
import numpy as np
import pickle

# ================================
# LOAD FILES
# ================================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# ================================
# CREATE APP
# ================================
app = FastAPI()

# ================================
# HOME ROUTE
# ================================
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running 🚀"}

# ================================
# PREDICTION ROUTE
# ================================
@app.post("/predict")
def predict(data: dict):
    try:
        input_data = data.copy()

        # ================================
        # APPLY LABEL ENCODING
        # ================================
        for col, le in le_dict.items():
            if col in input_data:
                input_data[col] = le.transform([input_data[col]])[0]

        # ================================
        # ENSURE CORRECT ORDER OF FEATURES
        # ================================
        features = [input_data[col] for col in feature_names]

        # Convert to numpy
        features_array = np.array([features])

        # ================================
        # SCALING
        # ================================
        scaled_data = scaler.transform(features_array)

        # ================================
        # PREDICTION
        # ================================
        prediction = model.predict(scaled_data)[0]

        return {
            "prediction": int(prediction),
            "result": "Default" if prediction == 1 else "No Default"
        }

    except Exception as e:
        return {"error": str(e)}