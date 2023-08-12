import pandas as pd
import numpy as np
import re
import sklearn
from zhconv import convert


def txt_processing(f):
    """
    txt处理
      1. 删除特定字符
      2. 繁转简
      3. 去重
    """
    text = []
    for line in f.readlines():
        line = line.replace('\n', '')  # 删除换行符
        line = convert(line, 'zh-cn')  # 繁转简
        text.append(line)

    text_02 = list(set(text))  # 去重
    result = '\n'.join(text_02)

    return result


# f = open('Data/other.txt', 'r', encoding='utf-8')
# result = txt_processing(f)
#
# # 保存文本
# with open('Data/new_other.txt', 'w', encoding='utf-8') as f:
#     f.write(result)


def merge_txt(path_01, path_02):
    """
    合并txt
    """
    text_01 = open(path_01, 'r', encoding='utf-8').read()
    text_02 = open(path_02, 'r', encoding='utf-8').read()

    text = text_01 + '\n' + text_02

    return text


# path_01 = 'name.txt'
# path_02 = 'other.txt'
# text = merge_txt(path_01, path_02)
#
# # 保存文本
# with open('Data.txt', 'w', encoding='utf-8') as f:
#     f.write(text)


def txt_to_csv(path):
    """
    txt转为csv
    """
    df = pd.read_csv(path, delimiter="\t")  # read_csv读取txt档

    # 栏位信息处理
    name = ['name']
    df.columns = name
    df['label'] = 0

    # 转csv
    df.to_csv("test.csv", encoding='utf-8-sig', index=False)


# path = 'new_other.txt'
# txt_to_csv(path)


def merge_csv(path_01, path_02):
    """
    合并dataframe
    """
    df_01 = pd.read_csv(path_01)
    df_02 = pd.read_csv(path_02)

    dataframe = pd.concat([df_01, df_02], axis=0)  # 合并
    dataframe = sklearn.utils.shuffle(dataframe)  # 合并后打乱

    dataframe.to_csv('dataframe.csv', encoding='utf-8-sig', index=False)


# path_01 = 'name_dataframe.csv'
# path_02 = 'other_dataframe.csv'
# merge_csv(path_01, path_02)


def list_to_csv():
    """
    list转为csv
    """
    list = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]

    name = ['name_01', 'name_02', 'name_03', 'name_04']
    dataframe = pd.DataFrame(columns=name, data=list)

    dataframe['label'] = 1
    print(dataframe)

    dataframe.to_csv("Data/dataframe.csv", index=False, encoding='utf-8-sig')


def remove_punctuation(line):
    """删除除字母、数字、汉字以外的所有符号"""

    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)

    return line
