from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="AeroGuard AI API", description="Production Backend for React UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Deep Learning Engine...")
try:
    model = load_model('aeroguard_lstm_engine.keras')
    scaler = joblib.load('aqi_pipeline_scaler.pkl')
    print("Engine Online and Ready.")
except Exception as e:
    print(f" ERROR LOADING FILES: Make sure .keras and .pkl are in the same folder as main.py. Details: {e}")

class SensorData(BaseModel):
    traffic: float
    wind: float
    factory: float
    temp: float

@app.get("/")
def health_check():
    """If you visit the Render URL in your browser, you should see this message."""
    return {"status": "online", "message": "AeroGuard AI Engine is actively listening."}
@app.post("/predict")
def predict_aqi(data: SensorData):
    try:
        sequence = np.zeros((24, 5))
        sequence[:, 0] = data.traffic
        sequence[:, 1] = data.wind
        sequence[:, 2] = data.factory
        sequence[:, 3] = data.temp

        sequence[:, 4] = (data.traffic * 1.4) + (data.factory * 1.1) - (data.wind * 1.6) + (data.temp * 0.3) + 30
   
        scaled_sequence = scaler.transform(sequence)
        ai_input = scaled_sequence.reshape(1, 24, 5)
        raw_prediction = model.predict(ai_input, verbose=0)
        dummy_array = np.zeros((1, 5))
        dummy_array[0, 4] = raw_prediction[0][0] 
        final_human_aqi = scaler.inverse_transform(dummy_array)[0, 4]

        return {
            "status": "success", 
            "predicted_aqi": round(final_human_aqi, 1)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))