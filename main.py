from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Carregar o modelo e o label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = FastAPI()

# Classe para definir a estrutura dos dados de entrada
class PumpData(BaseModel):
    soilMoisture: int
    temperature: int
    airMoisture: int

@app.post("/predict")
def predict(data: PumpData):
    try:
        # Criar um DataFrame com os dados de entrada
        input_data = pd.DataFrame({
            'soilMoisture': [data.soilMoisture],
            'temperature': [data.temperature],
            'airMoisture': [data.airMoisture]
        })

        # Fazer a previsão
        prediction = model.predict(input_data)

        # Converter a previsão para 'yes' ou 'no'
        result = label_encoder.inverse_transform(prediction)

        return {"activatePump": result[0]}



