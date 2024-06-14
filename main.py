from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Text
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs
import uvicorn
import json
import os

# Load and preprocess data
# with open('/content/merged_ta_data.json', 'r') as file:
#     ta_data = json.load(file)

with open('dataset/merged_hotel_data.json', 'r') as file:
    hotel_data = json.load(file)
    
user_data = pd.read_csv('dataset/user_with_preferences.csv')

ratings = [hotel['rating'] for hotel in hotel_data]
ratings_array = [[rating] for rating in ratings]

scaler = StandardScaler()
scaled_ratings = scaler.fit_transform(ratings_array)

for i, hotel in enumerate(hotel_data):
    hotel['scaled_rating'] = scaled_ratings[i][0]

mlb = MultiLabelBinarizer()
user_data['Preferences'] = user_data['Preferences'].apply(eval)  # Convert string to list
preferences_encoded = mlb.fit_transform(user_data['Preferences'])
preferences_df = pd.DataFrame(preferences_encoded, columns=mlb.classes_)
user_data = pd.concat([user_data, preferences_df], axis=1)

def calculate_middle_point(tour_interests):
    if not tour_interests:
        return {'lat': None, 'lng': None}
    lat_sum = sum(point['lat'] for point in tour_interests)
    lng_sum = sum(point['lng'] for point in tour_interests)
    middle_lat = lat_sum / len(tour_interests)
    middle_lng = lng_sum / len(tour_interests)
    return {'lat': middle_lat, 'lng': middle_lng}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lat2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

class RankingModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Hotel embeddings
        self.hotel_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=hotel_data["name"].unique(), mask_token=None),
            tf.keras.layers.Embedding(len(hotel_data["name"].unique()) + 1, embedding_dimension)
        ])

        # User preference embeddings
        self.user_preference_embeddings = tf.keras.layers.Embedding(len(mlb.classes_) + 1, embedding_dimension)

        # Rating prediction
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        # Task
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        hotel_embeddings = self.hotel_embeddings(features["hotel_name"])
        distance_normalized_expanded = tf.expand_dims(features["distance_normalized"], axis=1)
        rating_expanded = tf.expand_dims(features["rating"], axis=1)
        user_preferences = tf.reduce_sum(self.user_preference_embeddings(features["preferences"]), axis=1)
        concatenated = tf.concat([hotel_embeddings, distance_normalized_expanded, rating_expanded, user_preferences], axis=1)

        return self.rating_model(concatenated)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("ranking_score")
        predictions = self(features)
        return self.task(labels=labels, predictions=predictions)

with open('model/model_architecture.json', 'r') as f:
    model_json = f.read()
    model = tf.keras.models.model_from_json(model_json, custom_objects={'RankingModel': RankingModel})

dummy_input = {
    "hotel_name": tf.constant(["dummy_hotel"]),
    "distance_normalized": tf.constant([-0.041947]),
    "rating": tf.constant([4.0]),
    "preferences": tf.constant([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
}
_ = model(dummy_input)

model = RankingModel()
model.load_weights('model/model_weights_stdrecommend.h5')

def rank_hotels(user_id, top_n, tour_interests):
    middle_point = calculate_middle_point(tour_interests)
    hotel_data['distance'] = hotel_data.apply(lambda row: haversine(middle_point['lat'], middle_point['lng'], row['lat'], row['lng']), axis=1)
    hotel_data['distance_normalized'] = scaler.fit_transform(hotel_data[['distance']])
    
    user_preferences = user_data.loc[user_data['User_Id'] == user_id, mlb.classes_].values.flatten()
    hotel_data = {
        "hotel_name": tf.constant(hotel_data["name"].tolist()),
        "distance_normalized": tf.constant(hotel_data["distance_normalized"].tolist(), dtype=tf.float32),
        "rating": tf.constant(hotel_data["rating"].tolist(), dtype=tf.float32),
        "preferences": tf.constant([user_preferences] * len(hotel_data), dtype=tf.float32)
    }
    predictions = model(hotel_data)
    hotel_data["predicted_ranking_score"] = predictions.numpy().flatten()
    ranked_hotels = hotel_data.sort_values(by="predicted_ranking_score", ascending=False).head(top_n)
    return ranked_hotels[["name", "formatted_address", "distance", "rating", "predicted_ranking_score", "photos"]]

class RankRequestModel(BaseModel):
    user_id: int
    top_n: int
    tour_interests: List[Dict[str, float]]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/rank_hotels/")
async def rank_hotels_endpoint(request: RankRequestModel):
    try:
        result = rank_hotels(request.user_id, request.top_n, request.tour_interests)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Wander Rankings API!"}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5050))
    uvicorn.run(app, host="0.0.0.0", port=port)
