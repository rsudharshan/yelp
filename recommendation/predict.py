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

model.load_weights('rec_weights.h5')


def predict_rating(user_id, rest_id):
    if user_id <= N_USERS and rest_id <= N_RESTAURANTS:
        return model.predict([np.array([user_id-1]), np.array([rest_id-1])])[0][0]
    else:
        print("Enter user_id and rest_id withing scope, input Out of range")

predict_rating(200,1000)

user_ratings = stars88[stars88['ueid'] == TEST_USER][['user_id', 'business_id', 'stars']]
user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(1000, biz_to_beid[x['business_id']]), axis=1)
user_ratings