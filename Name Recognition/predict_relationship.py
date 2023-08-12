from tensorflow.keras import *
import pickle
from gensim.models import Word2Vec

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
    pred = model.predict(padded)
    cat_id = pred.argmax(axis=1)[0]

    return cat_id


# 模型文件名
model_w2v_file = r'Word2vec_Model/DaFengDaGengRen.model'
# 加载模型
model_w2v = Word2Vec.load(model_w2v_file)

# 获取输入词的相似词
enter_word = input("请输入预测的名字：")
result = model_w2v.wv.most_similar(enter_word)

for i in range(len(result)):
    pred_word = result[i][0]
    pred = predict(pred_word)
    if pred == 0:
        continue
    if pred == 1:
        print('与%s关系最近的是%s。' % (enter_word, pred_word))
        break
