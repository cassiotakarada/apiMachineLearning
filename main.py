from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Load the model and the label encoder using environment variables
model_path = os.getenv('MODEL_PATH', 'model.pkl')
label_encoder_path = os.getenv('LABEL_ENCODER_PATH', 'label_encoder.pkl')

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

app = FastAPI()

# Define the input data structure
class PumpData(BaseModel):
    soilMoisture: int
    temperature: int
    airMoisture: int

@app.post("/predict")
def predict(data: PumpData):
    try:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'soilMoisture': [data.soilMoisture],
            'temperature': [data.temperature],
            'airMoisture': [data.airMoisture]
        })

        # Make a prediction
        prediction = model.predict(input_data)

        # Convert the prediction to 'yes' or 'no'
        result = label_encoder.inverse_transform(prediction)

        return {"activatePump": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
