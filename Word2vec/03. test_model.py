from gensim.models import Word2Vec


def test_DaFeng_model():
    # 模型文件名
    model_file = r'word2vec_model/DaFengDaGengRen.model'
    # 加载模型
    model = Word2Vec.load(model_file)

    a = model.wv.index_to_key
    print(a)

    # 获取词的相似词
    # result = model.wv.most_similar('许七安', topn=100)
    # for word in result:
    #     print(word)
    # print()

    # 获取两个词之间的余弦相似度
    # result = model.wv.similarity('张无忌', '一')
    # print('相似程度:', result)


test_DaFeng_model()
