# -*- coding:utf-8 -*-
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import jieba
import time
import pickle
from lib import *
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tensorflow.keras import *

w2v_model_file = ''
log_line_num = 0


class Home():
    def __init__(self, master):
        self.root = master
        self.root.config()
        self.root.title('关系识别.net')
        self.root.geometry('800x500')  # 窗口大小
        self.root.iconbitmap('favicon.ico')  # 网页图标
        self.root['background'] = '#000000'

        window_one(self.root)


class window_one():
    def __init__(self, master):
        self.master = master
        self.master.config(bg='#000000')
        self.frame1 = tk.Frame(self.master, width=800, height=500, bg='#000000')
        self.frame1.pack()

        upload_label = tk.Label(self.frame1, text='文件上传', font=('楷体', 18), fg='#FFFFFF', bg="#000000")
        upload_label.place(relx=0.5, rely=0.3, anchor='center')
        upload_label_tip = tk.Label(self.frame1, text='（请上传TXT文件）', font=('楷体', 10), fg='#FFFFFF', bg="#000000")
        upload_label_tip.place(relx=0.5, rely=0.35, anchor='center')

        # 上传文件
        btn_update = tk.Button(self.frame1, text='上传', width=10, font=('楷体', 12), fg='#FFFFFF', bg="#222222",
                               command=self.upload_file)
        btn_update.place(relx=0.35, rely=0.5, anchor='center')
        self.entry_file = tk.Entry(self.frame1, width='30')
        self.entry_file.place(relx=0.55, rely=0.5, anchor='center')

        # 上传成功的提示
        self.tip_text = tk.StringVar()
        tip_label = tk.Label(self.frame1, textvariable=self.tip_text, font=('楷体', 12), fg='#FFFFFF', bg="#000000")
        tip_label.place(relx=0.5, rely=0.6, anchor='center')

    def upload_file(self):
        """上传文件"""

        global w2v_model_file
        file_path = tk.filedialog.askopenfilename()
        self.tip_text.set('上传中…')
        self.entry_file.insert(0, file_path)
        if file_path is not None:
            output_file = r'Data/' + file_path.split('/')[-1].split('.')[-2] + '.txt'
            with open(output_file, mode='w', encoding='utf-8') as output_f:
                with open(file=file_path, mode='r+', encoding='utf-8') as input_f:
                    for line in input_f.readlines():
                        words = jieba.cut(line.strip(), cut_all=False)  # jieba分词
                        output_f.write(' '.join(words) + '\n')  # 使用空格分隔并保存
        self.tip_text.set('上传成功')

        # 训练模型
        model = Word2Vec(LineSentence(output_file), vector_size=100)
        # 保存模型
        model_file = r'Word2vec_Model/' + file_path.split('/')[-1].split('.')[-2] + '.model'
        w2v_model_file = model_file
        model.save(model_file)

        for i in range(4):
            time.sleep(1)
            if i == 3:
                self.frame1.destroy()
                window_two(self.master)


class window_two():
    def __init__(self, master):
        global w2v_model_file
        self.master = master
        self.master.config(bg='#000000')
        self.frame2 = tk.Frame(self.master, width=800, height=500, bg='#000000')
        self.frame2.pack()

        myName_label = tk.Label(self.frame2, text='分析的名字:', font=('楷体', 15), fg='#FFFFFF', bg="#000000")
        myName_label.place(relx=0.1, rely=0.06, anchor='center')
        self.myName_text = tk.Text(self.frame2, width=10, height=1, font='楷体', fg='#C0C0C0', bg="#555555",
                                   relief='sunken')
        self.myName_text.place(relx=0.24, rely=0.06, anchor='center')

        self.sex_label = tk.Label(self.frame2, text='对方性别:', font=('楷体', 15), fg='#FFFFFF', bg="#000000")
        self.sex_label.place(relx=0.42, rely=0.06, anchor='center')

        # 复选框的样式设置
        self.frame2.option_add("*TCombobox*Font", ('楷体', 12))
        self.frame2.option_add("*TCombobox*Background", "#555555")
        self.frame2.option_add("*TCombobox*Foreground", "#F5DEB3")
        Style = ttk.Style()
        Style.theme_create('Style', settings={'TCombobox': {
            'configure': {'padding': 1, 'foreground': '#C0C0C0', 'background': '#555555', 'selectbackground': '#555555',
                          'fieldbackground': '#555555'}}})
        Style.theme_use('Style')

        self.sex_cbox = ttk.Combobox(self.frame2, width=10, height=10, font='楷体', background='black',
                                     values=['不限', '男生', '女生'])
        self.sex_cbox.current(0)
        self.sex_cbox.place(relx=0.55, rely=0.06, anchor='center')

        btn_pred = tk.Button(self.frame2, text='开始分析', font=('楷体', 15), fg='#FFFFFF', bg="#111111", command=self.pred)
        btn_pred.place(relx=0.75, rely=0.06, anchor='center')

        # 结果展示框
        self.show_command = tk.Text(self.frame2, width=50, height=22, font='楷体', fg='#FFE4E1', bg='#303030')
        self.show_command.place(relx=0.28, rely=0.5, anchor='center')

        # 日志框
        self.log_data_Text = tk.Text(self.frame2, width=38, height=10, font='楷体', fg='#FFE4E1', bg='#303030')
        self.log_data_Text.place(relx=0.77, rely=0.69, anchor='center')

    def pred(self):
        # 加载预测模型
        model = models.load_model('Model/name-recognition_model_sex.h5')

        # 加载tokenizer
        with open('Tokenizer/tokenizer_sex.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # 加载w2v模型
        model_w2v = Word2Vec.load(w2v_model_file)

        # 取得输入词
        enter_word = self.myName_text.get("1.0", "end")[:-1]

        # 判断输入词在不在文章词库中
        word_in_essay = enter_word in model_w2v.wv.index_to_key
        if word_in_essay == False:
            self.write_log_to_Text("INFO: 抱歉！本文查无此人！")

        # 获取输入词的相似词
        result = model_w2v.wv.most_similar(enter_word)

        if self.sex_cbox.get() == '不限':
            for i in range(len(result)):
                pred_word = result[i][0]  # 名字
                intimacy = result[i][1] * 100  # 亲密度
                pred = predict(pred_word, tokenizer, model)
                if pred == 2:
                    if i == len(result) - 1:
                        self.write_log_to_Text("INFO: 抱歉！没有找到与他相关的人物！")
                    continue
                if pred == 0 or pred == 1:
                    self.show_command.insert('end', '与%s关系最近的是%s。亲密度为%d。\n\n' % (enter_word, pred_word, intimacy))
                    self.write_log_to_Text("INFO: 查询完成！")
                    break

        if self.sex_cbox.get() == '男生':
            for i in range(len(result)):
                pred_word = result[i][0]
                intimacy = result[i][1] * 100
                pred = predict(pred_word, tokenizer, model)
                if pred == 2 or pred == 0:
                    if i == len(result) - 1:
                        self.write_log_to_Text("INFO: 抱歉！没有找到与他相关的男生！")
                    continue
                if pred == 1:
                    self.show_command.insert('end', '与%s关系最近的是%s。亲密度为%d。\n\n' % (enter_word, pred_word, intimacy))
                    self.write_log_to_Text("INFO: 查询完成！")
                    break

        if self.sex_cbox.get() == '女生':
            for i in range(len(result)):
                pred_word = result[i][0]
                intimacy = result[i][1] * 100
                pred = predict(pred_word, tokenizer, model)
                if pred == 2 or pred == 1:
                    if i == len(result) - 1:
                        self.write_log_to_Text("INFO: 抱歉！没有找到与他相关的女生！")
                    continue
                if pred == 0:
                    self.show_command.insert('end', '与%s关系最近的是%s。亲密度为%d。\n\n' % (enter_word, pred_word, intimacy))
                    self.write_log_to_Text("INFO: 查询完成！")
                    break

    def get_current_time(self):
        """获取当前时间"""

        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        return current_time

    def write_log_to_Text(self, logmsg):
        """日志动态打印"""

        global log_line_num
        current_time = self.get_current_time()
        logmsg_in = str(current_time) + " " + str(logmsg) + "\n\n"  # 日志内容

        # 日志不超过5条
        if log_line_num < 5:
            self.log_data_Text.insert("end", logmsg_in)
            log_line_num = log_line_num + 1
        else:
            self.log_data_Text.delete("1.0", "3.0")
            self.log_data_Text.insert("end", logmsg_in)


if __name__ == '__main__':
    root = tk.Tk()
    Home(root)
    root.mainloop()
