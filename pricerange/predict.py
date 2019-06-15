import pandas as pd
import numpy as np
import tensorflow as tf
import re
from numpy import argmax

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import Constant
import pickle 


replace_puncts = {'`': "'", '′': "'", '“':'"', '”': '"', '‘': "'"}

strip_chars = [',', '.', '"', ':', ')', '(', '-', '|', ';', "'", '[', ']', '>', '=', '+', '\\', '•',  '~', '@', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

puncts = ['!', '?', '$', '&', '/', '%', '#', '*','£']
sequence_length = 300
max_features = 20000
num_words = 20001
embedding_dim = 100
with open("embedding_matrix.pkl", "rb") as f:
    embedding_matrix = pickle.load(f) 

def clean_str(x):
    x = str(x)
    x = x.lower()
    x = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "url", x)
    for k, v in replace_puncts.items():
        x = x.replace(k, f' {v} ')
    for punct in strip_chars:
        x = x.replace(punct, ' ') 
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    x = x.replace(" '", " ")
    x = x.replace("' ", " ")
    return x


model = tf.keras.models.Sequential()  
def build_model():
    model.add(Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=sequence_length))
    model.add(SpatialDropout1D(0.2))

    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.25))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.load_weights("model1.h5")
    model.summary()



def process_reviews(reviews):
    for i,j in enumerate(reviews):
        reviews[i] = clean_str(j)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, split=' ', oov_token='<unw>', filters=' ')
    tokenizer.fit_on_texts(reviews)
    X = tokenizer.texts_to_sequences(reviews)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, sequence_length)
    return X

def predict_price(reviews):
    X = process_reviews(reviews)
    N = len(reviews)
    a = np.zeros(4)
    for i in range(N):
        y = model.predict(x=tf.expand_dims(X[i], 0),steps=1, verbose=1)
        a+=y[0]
    a = a/N
    print(a)
    classes = np.argmax(a, axis=0)
    return classes+1



build_model()


rev = ["""After watching a ton of his shows on TV, I've been wanting to try some of his signature items to see if he lives up to his reputation.

This is one of his Vegas flagships. It was my first time here but hubby had been here before and loved it. Trendy decor and cute story about going through the Chunnel from Paris to London when you're being seated.

Excellent service at the table! The hostesses aren't the most consistent, but the table service is excellent. Nice explanation of the menu and steak cuts.

These are the highlights and what I would definitely recommend getting:
- Beef Wellington - it's the signature item and sooo good. The layers are perfect and the meat inside is incredibly flavorful
- Sticky toffee pudding - another classic dish that totally lived up to the hype. Definitely good for sharing after a big meal.

Bread service is lovely with a nice variety to choose from, and we also enjoyed the salad to start (not unique, but a nice starter if you need vegetables).  The Mac and cheese side is totall…""" , 
"Great_hotel"
                ]
print(predict_price(rev))

