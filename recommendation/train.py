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


import numpy as np
from keras.layers import Embedding, Reshape, Merge
from keras.models import Sequential

class CFModel(Sequential):
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))

        super(CFModel, self).__init__(**kwargs)
        
        # The Merge layer takes the dot product of user and movie latent factor vectors to return the corresponding rating.
        self.add(Merge([P, Q], mode='dot', dot_axes=1))

    # The rate function to predict user's rating of unrated items
    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]


model = CFModel(N_USERS+1, N_RESTAURANTS+1, K_FACTORS)
model.compile(loss='mse', optimizer='adamax')

# Save the model weights each time the validation loss has improved
callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint('rec_weights.h5', save_best_only=True)]

# Use 30 epochs, 90% training data, 10% validation data 
history = model.fit([Users, Restaurants], Stars, nb_epoch=50, validation_split=.1, verbose=2, callbacks=callbacks)

import pandas 
df = pandas.DataFrame(stars88["business_id"])
df["beid"]=stars88["beid"]
biz_to_beid = {}
for i,row in df.iterrows():
    if row["business_id"] not in biz_to_beid:
        biz_to_beid[row["business_id"]]= row["beid"]
beid_to_biz = dict(map(reversed, biz_to_beid.items()))

import pickle
with open('id_store.pkl', 'w') as f:
    pickle.dump([biz_to_beid, beid_to_biz, user_to_ueid,ueid_to_user], f,protocol=-1)

f.close()

