from fastapi import FastAPI
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.utils import get_file 
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps
from typing import List
from uvicorn import run
from utils import load_model
import os

app = FastAPI()
model = load_model("model/food-vision-model.h5")

class UserPreferences(BaseModel):
    location: List[float]
    preferences: List[str]
    
class Hotel(BaseModel):
    name: str
    location: List[float]
    preferences: List[str]

class HotelRecommendation(BaseModel):
    hotel: List[Hotel]

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Wander Rankings API!"}
 
@app.post("/net/image/prediction/")
async def get_net_image_prediction():
    if image_link == "":
        return {"message": "No image link provided"}
    
    img_path = get_file(
        origin = image_link
    )
    img = load_img(
        img_path, 
        target_size = (224, 224)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    pred = model.predict(img_array)
    score = softmax(pred[0])

    class_prediction = class_predictions[argmax(score)]
    model_score = round(max(score) * 100, 2)
    model_score = dumps(model_score.tolist())

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score
    }


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5050))
	run(app, host="0.0.0.0", port=port)