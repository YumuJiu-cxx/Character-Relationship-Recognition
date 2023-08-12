import pandas as pd
import numpy as np
import re
import os
import pickle
from lib import *

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

df = pd.read_csv('Data/dataframe_sex.csv')
print(df, '\n')

df['clean_name'] = df['name'].apply(remove_punctuation)  # 删除除字母、数字、汉字以外的所有符号
df['cut_name'] = df['clean_name'].apply(lambda x: " ".join([w for w in x]))  # 分词
print(df, '\n')

"""
  LSTM建模
"""
# 定义常量
max_nb_word = 50000  # 设置最频繁使用的50000个词
max_sequence_length = 10  # 设置每条df['clean_review']最大的长度
embedding_dim = 256  # 设置Embedding层的维度

tokenizer = Tokenizer(num_words=max_nb_word)
tokenizer.fit_on_texts(df['cut_name'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index), '\n')

# 保存tokenizer
with open('Tokenizer/tokenizer_sex.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X = tokenizer.texts_to_sequences(df['cut_name'].values)
X = pad_sequences(X, maxlen=max_sequence_length)  # 填充X,让X的各个列的长度统一

# 多类标签的onehot展开
Y = pd.get_dummies(df['label']).values
# Y = df['label']

print(X.shape)
print(Y.shape, '\n')

# 定义模型
model = Sequential()
model.add(keras.layers.Embedding(max_nb_word, embedding_dim, input_length=X.shape[1]))
model.add(keras.layers.SpatialDropout1D(0.2))
model.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 加载model参数，继续训练
# input_dir = "checkpoints"
# model.load_weights(tf.train.latest_checkpoint(input_dir))  # 加载model的权重

"""
  callback模块-checkpoints
"""
output_dir = "checkpoints"
if not os.path.exists(output_dir):  # 如果没有此文件夹，则新建
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')  # 连接两个路径名组件，./text_generation_checkpoints/ckpt_{epoch}
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)  # 保存模型权重用于回调

"""
  训练模型
"""
epochs = 1
batch_size = 64

# 训练模型
history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint_callback])
# 自动找到最近保存的变量文件
new_checkpoint = tf.train.latest_checkpoint(output_dir)

# 保存模型
model.save('Model/name-recognition_model_sex.h5')
