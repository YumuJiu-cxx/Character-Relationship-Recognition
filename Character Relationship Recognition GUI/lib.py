import re
from keras.preprocessing.sequence import pad_sequences


def remove_punctuation(line):
    """删除除字母、数字、汉字以外的所有符号"""

    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)

    return line


def predict(text, tokenizer, model):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in txt])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=10)
    pred = model.predict(padded)
    name_or_not = pred.argmax(axis=1)[0]

    return name_or_not
