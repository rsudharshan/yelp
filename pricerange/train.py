import pandas as pd
import numpy as np
#import seaborn as sns
import tensorflow as tf
import re
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.initializers import Constant


reviews = pd.read_csv("bq-100k-reviews.csv")

import re 

replace_puncts = {'`': "'", '′': "'", '“':'"', '”': '"', '‘': "'"}

strip_chars = [',', '.', '"', ':', ')', '(', '-', '|', ';', "'", '[', ']', '>', '=', '+', '\\', '•',  '~', '@', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

puncts = ['!', '?', '$', '&', '/', '%', '#', '*','£']

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

reviews['processed'] = reviews['text'].apply(clean_str)


sequence_length = 300
max_features = 20000 # this is the number of words we care about

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(reviews['processed'].values)

# this takes our sentences and replaces each word with an integer
X = tokenizer.texts_to_sequences(reviews['processed'].values)

# we then pad the sequences so they're all the same length (sequence_length)
X = tf.keras.preprocessing.sequence.pad_sequences(X, sequence_length)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

y = pd.get_dummies(reviews['RestaurantsPriceRange2']).astype(float).values

# lets keep a couple of thousand samples back as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("test set size " + str(len(X_test)))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

import os 
embeddings_index = {}
f = open(os.path.join('data/glove.6B', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 100

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)

import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


model = tf.keras.models.Sequential()

model.add(Embedding(num_words,
                   embedding_dim,
                   embeddings_initializer=Constant(embedding_matrix.tolist()),
                   input_length=sequence_length,
                   trainable=True))
model.add(SpatialDropout1D(0.2))

model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))

model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.25))
model.add(Dense(units=4, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

# define the checkpoint
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

batch_size = 128
history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=1, validation_split=0.1, callbacks=callbacks_list)

#Save entire model to a HDF5 file
model.save('model.h5')

# model = tf.keras.models.load_model('model.h5')
# h = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=1, validation_split=0.1, callbacks=callbacks_list)
# model.evaluate(x=X_test, y=y_test, batch_size=batch_size, verbose=1)

