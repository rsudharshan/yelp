import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

stars88 = pd.read_csv('stars_88k_train.csv')

N_USERS = stars88['ueid'].drop_duplicates().max()
N_RESTAURANTS = stars88['beid'].drop_duplicates().max()

print(N_USERS, " Users")
print(N_RESTAURANTS, " restaurants")

RNG_SEED=3
# Create training set
shuffled_ratings = stars88.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['ueid'].values
print('Users:', Users, ', shape =', Users.shape)
Restaurants = shuffled_ratings['beid'].values
print('Restaurants:', Restaurants, ', shape =', Restaurants.shape)
Stars = shuffled_ratings['stars'].values
print('Stars:', Stars, ', shape =', Stars.shape)


K_FACTORS = 100 
TEST_USER = 200 


from cfmodel import CFModel
model = CFModel(N_USERS+1, N_RESTAURANTS+1, K_FACTORS)
model.compile(loss='mse', optimizer='adamax')

# Save the model weights each time the validation loss has improved
callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint('rec_weights.h5', save_best_only=True)]

history = model.fit([Users, Restaurants], Stars, nb_epoch=5, validation_split=.1, verbose=2, callbacks=callbacks)

import pandas 
df = pandas.DataFrame(stars88["business_id"])
df["beid"]=stars88["beid"]
biz_to_beid = {}
for i,row in df.iterrows():
    if row["business_id"] not in biz_to_beid:
        biz_to_beid[row["business_id"]]= row["beid"]
beid_to_biz = dict(map(reversed, biz_to_beid.items()))

df = pandas.DataFrame(stars88["user_id"])
df["ueid"]=stars88["ueid"]
user_to_ueid = {}
for i,row in df.iterrows():
    if row["user_id"] not in user_to_ueid:
        user_to_ueid[row["user_id"]]= row["ueid"]
ueid_to_user = dict(map(reversed, user_to_ueid.items()))

import pickle
with open('id_store.pkl', 'w') as f:
    pickle.dump([biz_to_beid, beid_to_biz, user_to_ueid,ueid_to_user, N_USERS, N_RESTAURANTS, K_FACTORS], f,protocol=-1)

f.close()

