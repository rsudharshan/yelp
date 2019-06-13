import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stars = pd.read_csv('stars.csv',usecols=['user_id','ueid','business_id','beid', 'stars'])
N_USERS = stars['ueid'].drop_duplicates().max()
N_RESTAURANTS = stars['beid'].drop_duplicates().max()

print(N_USERS, " Users")
print(N_RESTAURANTS, " restaurants")

RNG_SEED = 9
randomized_stars = stars.sample(frac=1., random_state=RNG_SEED)
users = randomized_stars['ueid'].values
restaurants = randomized_stars['beid'].values
stars = randomized_stars['stars'].values

K_FACTORS = 100 
TEST_USER = 200 

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Reshape, Merge
from keras.models import Sequential

model = Sequential()
P = Sequential()
P.add(Embedding(N_USERS, K_FACTORS, input_length=1))
P.add(Reshape((K_FACTORS,)))
Q = Sequential()
Q.add(Embedding(N_RESTAURANTS, K_FACTORS, input_length=1))
Q.add(Reshape((K_FACTORS,)))
model.add(Merge([P, Q], mode='dot', dot_axes=1))
model.compile(loss='mse', optimizer='adamax')

# Save the model weights each time the validation loss has improved
callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint('rec_weights.h5', save_best_only=True)]

# Use 30 epochs, 90% training data, 10% validation data 
history = model.fit([users, restaurants], stars, nb_epoch=30, validation_split=.1, verbose=2, callbacks=callbacks)
