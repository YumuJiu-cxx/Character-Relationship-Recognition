import jieba


class PreProcessor:
    def text_segment(self, input_file, output_file):
        with open(output_file, mode='w', encoding='utf-8') as output_f:
            with open(input_file, mode='r', encoding='utf-8') as input_f:
                index = 0
                for line in input_f.readlines():
                    # 对文本进行分词操作
                    words = jieba.cut(line.strip(), cut_all=False)
                    # 将词之间使用空格分隔，并保存到文件中
                    output_f.write(' '.join(words) + '\n')
                    index += 1
                    # 打印处理进度
                    if index % 1000 == 0:
                        print("Segment text {} lines...".format(index))


input_text_file = r'../Novel-Text/YiTianTuLongJi.txt'
output_text_file = r'text/YiTianTuLongJi' + '_segment.txt'

# 对文本进行分词操作
preprocessor = PreProcessor()
preprocessor.text_segment(input_text_file, output_text_file)
