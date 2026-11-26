import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title='Mobile Price Range Prediction API',
    description='A production-ready machine learning API predicting mobile phone price categories.',
    version='1.0.0',
)

# Enable CORS for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load the trained pipeline
try:
  model_pipeline = joblib.load('mobile_price_pipeline.pkl')
  print('Model pipeline loaded successfully.')
except Exception as e:
  print(f'Error loading model pipeline: {e}')


# Define Pydantic Input Schema matching dataset features exactly
class MobileInput(BaseModel):
  battery_power: int = Field(..., example=842)
  blue: int = Field(..., example=0)
  clock_speed: float = Field(..., example=2.2)
  dual_sim: int = Field(..., example=0)
  fc: int = Field(..., example=1)
  four_g: int = Field(..., example=0)
  int_memory: int = Field(..., example=7)
  m_dep: float = Field(..., example=0.6)
  mobile_wt: int = Field(..., example=188)
  n_cores: int = Field(..., example=2)
  pc: int = Field(..., example=2)
  px_height: int = Field(..., example=20)
  px_width: int = Field(..., example=756)
  ram: int = Field(..., example=2549)
  sc_h: int = Field(..., example=9)
  sc_w: int = Field(..., example=7)
  talk_time: int = Field(..., example=19)
  three_g: int = Field(..., example=0)
  touch_screen: int = Field(..., example=0)
  wifi: int = Field(..., example=1)


@app.get('/')
def home():
  return {
      'status': 'Online',
      'message': 'Mobile Price Classification API is running live!',
      'docs': '/docs',
  }


@app.post('/predict')
def predict_price(data: MobileInput):
  try:
    input_data = pd.DataFrame([data.dict()])
    prediction = model_pipeline.predict(input_data)
    probabilities = model_pipeline.predict_proba(input_data)

    price_labels = {
        0: 'Low Cost',
        1: 'Medium Cost',
        2: 'High Cost',
        3: 'Very High Cost',
    }
    predicted_class = int(prediction[0])
    confidence = float(np.max(probabilities) * 100)

    return {
        'predicted_price_range': predicted_class,
        'price_category': price_labels.get(predicted_class),
        'confidence_percentage': round(confidence, 2),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
