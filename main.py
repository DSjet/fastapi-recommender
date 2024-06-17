from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs
import uvicorn
import json
import os

# Load data
with open('dataset/merged_ta_data.json', 'r') as file:
    ta_data = json.load(file)
    
hotel = pd.read_csv('dataset/merged_hotels.csv')
user_data = pd.read_csv('dataset/user_with_preferences.csv')

# Ensure the 'name' column is of type str and remove duplicates
hotel['name'] = hotel['name'].astype(str)
hotel = hotel.drop_duplicates(subset=['name'])

# Scale ratings
ratings = hotel['rating'].values.reshape(-1, 1)
scaler = StandardScaler()
scaled_ratings = scaler.fit_transform(ratings)

hotel['scaled_rating'] = scaled_ratings

# Encode user preferences
mlb = MultiLabelBinarizer()
user_data['Preferences'] = user_data['Preferences'].apply(eval)  # Convert string to list
preferences_encoded = mlb.fit_transform(user_data['Preferences'])
preferences_df = pd.DataFrame(preferences_encoded, columns=mlb.classes_)
user_data = pd.concat([user_data, preferences_df], axis=1)

# Helper functions
def calculate_middle_point(tour_interests: List[Dict[str, float]]) -> Dict[str, float]:
    if not tour_interests:
        return {}
    lat_sum = sum(point['lat'] for point in tour_interests)
    lng_sum = sum(point['lng'] for point in tour_interests)
    middle_lat = lat_sum / len(tour_interests)
    middle_lng = lng_sum / len(tour_interests)
    return {'lat': middle_lat, 'lng': middle_lng}

ta_dict = {item['name']: {'lat': item['lat'], 'lng': item['lng']} for item in ta_data}
hotel_dict = {row['name']: {'lat': row['lat'], 'lng': row['lng']} for _, row in hotel.iterrows()}

def dptin_kordinat(name: str, data_dict: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
  return data_dict.get(name)

def get_coordinates(name: str, data_dict: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    return data_dict.get(name)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def compute_distances(middle_point: Dict[str, float], data: List[Dict[str, Any]]) -> np.ndarray:
    lat1, lon1 = middle_point['lat'], middle_point['lng']
    lat2 = np.array([item['lat'] for item in data])
    lon2 = np.array([item['lng'] for item in data])
    return haversine(lat1, lon1, lat2, lon2)

# Calculate initial distances and scale them
middle_point = calculate_middle_point([
    get_coordinates("Air Panas Semurup", ta_dict),
    get_coordinates("Tebat Air Koto Majidin", ta_dict),
    get_coordinates("Air Terjun Pendung Mudik", ta_dict),
    get_coordinates("Kebun jeruk arumi&hanum", ta_dict)
])

distances = compute_distances(middle_point, hotel.to_dict(orient='records'))
scaled_distances = scaler.fit_transform(distances.reshape(-1, 1))

hotel['distance_normalized'] = scaled_distances

# Define the ranking model
class RankingModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Hotel embeddings
        self.hotel_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=hotel['name'].tolist(), mask_token=None),
            tf.keras.layers.Embedding(len(hotel) + 1, embedding_dimension)
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

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        hotel_embeddings = self.hotel_embeddings(features["hotel_name"])
        distance_normalized_expanded = tf.expand_dims(features["distance_normalized"], axis=1)
        rating_expanded = tf.expand_dims(features["rating"], axis=1)
        user_preferences = tf.reduce_sum(self.user_preference_embeddings(features["preferences"]), axis=1)
        concatenated = tf.concat([hotel_embeddings, distance_normalized_expanded, rating_expanded, user_preferences], axis=1)

        return self.rating_model(concatenated)

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("ranking_score")
        predictions = self(features)
        return self.task(labels=labels, predictions=predictions)

# Create and build the model before loading weights
model = RankingModel()

# Use dummy input to build the model
dummy_input = {
    "hotel_name": tf.constant(["dummy_hotel"]),
    "distance_normalized": tf.constant([0.0], dtype=tf.float32),
    "rating": tf.constant([4.0], dtype=tf.float32),
    "preferences": tf.constant([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=tf.float32)
}
_ = model(dummy_input)

# Now load the weights
model.load_weights('model/model_weights_stdrecommend.h5')

def rank_hotels(user_id, top_n, tour_interests):
    try:
        # Calculate middle point of the tour interests
        middle_point = calculate_middle_point(tour_interests)

        # Compute distances from the middle point to all hotels
        hotel['distance'] = hotel.apply(lambda row: haversine(middle_point['lat'], middle_point['lng'], row['lat'], row['lng']), axis=1)
        hotel['distance_normalized'] = scaler.fit_transform(hotel[['distance']])

        # Get user preferences
        user_preferences = user_data.loc[user_data['User_Id'] == user_id, mlb.classes_].values.flatten()

        # Prepare features for the model
        hotel_features = {
            "hotel_name": tf.constant(hotel["name"].tolist(), dtype=tf.string),
            "distance_normalized": tf.constant(hotel["distance_normalized"].tolist(), dtype=tf.float32),
            "rating": tf.constant(hotel["rating"].tolist(), dtype=tf.float32),
            "preferences": tf.constant([user_preferences] * len(hotel), dtype=tf.float32)
        }

        # Make predictions
        predictions = model(hotel_features)

        # Add predicted ranking score to the hotel dataframe
        hotel["predicted_ranking_score"] = predictions.numpy().flatten()

        # Sort hotels by predicted ranking score and return top_n hotels
        ranked_hotels = hotel.sort_values(by="predicted_ranking_score", ascending=False).head(top_n)

        # Convert to standard Python types for JSON serialization
        def replace_invalid_values(value):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)

        ranked_hotels["distance"] = ranked_hotels["distance"].apply(replace_invalid_values)
        ranked_hotels["rating"] = ranked_hotels["rating"].apply(replace_invalid_values)
        ranked_hotels["predicted_ranking_score"] = ranked_hotels["predicted_ranking_score"].apply(replace_invalid_values)
        
        json_str = ranked_hotels[["name", "formatted_address", "distance", "rating", "predicted_ranking_score", "photos"]].to_json(orient="records", force_ascii=False)

        # Use json.dumps to ensure correct handling of backslashes
        json_str_fixed = json.dumps(json.loads(json_str), ensure_ascii=False)
        
        return json_str

    except Exception as e:
        # Print detailed error message for debugging
        print(f"Error in rank_hotels function: {str(e)}")
        raise e

class HotelRankRequestModel(BaseModel):
    user_id: int
    top_n: int
    tour_interests: List[str]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/rank_hotels/")
async def rank_hotels_endpoint(request: HotelRankRequestModel):
    try:
        print(f"Received request: {request}")
        print(f"Tour interests: {request.tour_interests}")
        
        tour_interests = request.tour_interests
        tour_interest_loc = [dptin_kordinat(point, ta_dict) for point in tour_interests]
        # tour_interest_loc = [dptin_kordinat(point, ta_dict) for point in request.tour_interests]
        # for point in request.tour_interests:
        #     tour_interest_loc.append(dptin_kordinat(point, ta_dict))
        
        result = rank_hotels(request.user_id, request.top_n, tour_interest_loc)
        print(result)
        
        result_json = json.loads(result)
        
        return result_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Wander Rankings API!"}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
