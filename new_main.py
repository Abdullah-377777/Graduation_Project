import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Load your trained model and scalers
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up static files and templates
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

# Get the API key from the environment variable
API_KEY = os.getenv("API_KEY")

# Define input data model to accept a list of floats
class InputData(BaseModel):
    data: list[float]  # Change to a list of floats

class PredictionResponse(BaseModel):
    prediction: str

# Set up API key security
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key")

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: InputData, api_key: str = Depends(verify_api_key)):
    try:
        # Scale the input data
        scaled_data = scaler.transform([input_data.data])  # Pass the entire list
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        # Decode the prediction
        decoded_prediction = label_encoder.inverse_transform(prediction)
        
        return PredictionResponse(prediction=decoded_prediction[0])
    except Exception as e:
        import traceback
        print("Prediction error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("Prediction.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})