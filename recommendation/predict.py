import numpy as np
from keras.layers import Embedding, Reshape, Merge
from keras.models import Sequential
import pickle

with open("id_store.pkl", "rb") as f:
    l = pickle.load(f) 
    biz_to_beid = l[0]
    beid_to_biz = l[1] 
    user_to_ueid = l[2] 
    ueid_to_user = l[3]
    N_USERS = l[4]
    N_RESTAURANTS = l[5]
    K_FACTORS = l[6]

import pandas as pd
stars = pd.read_csv('stars_88k_train.csv')
restaurants = pd.read_csv('restaurants.csv')

from cfmodel import CFModel 
model = CFModel(N_USERS+1, N_RESTAURANTS+1, K_FACTORS)
model.compile(loss='mse', optimizer='adamax')
model.load_weights('rec_weights.h5')


def predict_rating(user_id, rest_id):
    if user_id <= N_USERS and rest_id <= N_RESTAURANTS:
        return model.predict([np.array([user_id]), np.array([rest_id])])[0][0]
    else:
        print("Enter user_id and rest_id withing scope, input Out of range")

def recommend(user_id, city):
    user_ratings = stars[stars['ueid'] == user_id][['user_id', 'business_id', 'stars']]
    user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(user_id, biz_to_beid[x['business_id']]), axis=1)
    rec_stars = stars[stars['business_id'].isin(user_ratings['business_id']) == False][['business_id']].drop_duplicates() 
    city_filtered = restaurants[restaurants["city"]==city]
    recommendations = rec_stars.merge(city_filtered,on='business_id',how='inner') [["business_id","name","city"]]
    recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(user_id, biz_to_beid[x['business_id']]), axis=1)
    return recommendations.sort_values(by='prediction',ascending=False).head().to_json(orient="index")

print(recommend(500,"Las Vegas"))
from flask import Flask, request
app = Flask(__name__)

@app.route("/recommend", methods = ['GET'])
def rec_response():
    city = request.args.get('city')
    user_id = int(request.args.get('userid'))
    return recommend(user_id, city)