from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train_DaFeng():
    # 训练数据文件名
    data_file = r'text/YiTianTuLongJi_segment.txt'
    print(data_file)

    # 保存的模型文件名
    model_file = r'word2vec_model/YiTianTuLongJi.model'
    vector_file = r'word2vec_model/YiTianTuLongJi.vector'

    # 训练模型
    model = Word2Vec(LineSentence(data_file), vector_size=100)
    # 保存模型
    model.save(model_file)
    model.wv.save_word2vec_format(vector_file, binary=False)


train_DaFeng()
