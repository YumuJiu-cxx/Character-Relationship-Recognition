from tensorflow.keras import *
import numpy as np
import re
import pickle

from lib import *
from keras.preprocessing.sequence import pad_sequences

model = models.load_model('Model/name-recognition_model_sex.h5')

# 加载tokenizer
with open('Tokenizer/tokenizer_sex.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in txt])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=10)
    # cat_id = model.predict(padded)[0][0]
    pred = model.predict(padded)
    cat_id = pred.argmax(axis=1)[0]

    return cat_id


pred_word = '林樱'
pred = predict(pred_word)
print('是否为名字:', pred)
