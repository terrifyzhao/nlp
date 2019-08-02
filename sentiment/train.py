import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import json
from tqdm import tqdm
import os
import pandas as pd
from data_utils import *
from keras.callbacks import Callback
from keras_bert import Tokenizer
import codecs
import tensorflow as tf
from sentiment.sentiment_model import sentiment_model

dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'
max_len = 315
token_dict = {}
additional_chars = set()
global graph
graph = tf.get_default_graph()

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

length = 100


def read_data():
    df_train = pd.read_csv('../data/sentiment/train.csv', header=None).values[1:]
    df_eval = pd.read_csv('../data/sentiment/eval.csv', header=None).values[1:]
    df_test = pd.read_csv('../data/sentiment/test.csv', header=None).values[1:]

    df_train = shuffle(df_train)
    df_eval = shuffle(df_eval)
    df_test = shuffle(df_test)

    return df_train[0:length][0], df_eval[0:length][0], df_test[0:length][0]


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


def data_generator(data, batch_size):
    while True:
        X1, X2, Y = [], [], []
        for i, d in enumerate(data):
            x = d[0].strip()
            y = int(d[1])

            x = tokenizer.encode(first=x)
            X1.append(x[0])
            X2.append(x[1])
            Y.append(y)

            if len(X1) == batch_size or i == len(data) - 1:
                X1 = pad_sequences(X1, maxlen=max_len)
                X2 = pad_sequences(X2, maxlen=max_len)
                Y = one_hot(Y, 4)
                yield [X1, X2], Y
                X1, X2, Y = [], [], []


def extract_entity(text_in, model):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    text_in = text_in[:max_len]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    with graph.as_default():
        _ps1 = model.predict([_x1, _x2])
        start = _ps1.argmax()

    return start, _ps1


class Evaluate(Callback):
    def __init__(self, data, model):
        self.ACC = []
        self.best = 0.
        self.dev_data = data
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            self.model.save_weights('../output/sentiment_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        for d in tqdm(iter(self.dev_data)):
            R = extract_entity(d[0], self.model)
            if R == d[1]:
                A += 1
        return A / len(self.dev_data)


def test(test_data, model):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    with open('../result.txt', 'w', encoding='utf-8')as file:
        for d in tqdm(iter(test_data)):
            s = str(d[0]) + ',' + extract_entity(d[1].replace('\t', ''), model)
            file.write(s + '\n')


if __name__ == '__main__':
    batch_size = 8
    learning_rate = 1e-5
    is_test = True

    train_data, dev_data, test_data = read_data()

    model = sentiment_model(learning_rate=learning_rate)

    # model.load_weights('../output/best_model2.weights')

    evaluator = Evaluate(dev_data, model)

    X = data_generator(train_data, batch_size)
    steps = int((len(train_data) + batch_size - 1) / batch_size)

    model.fit_generator(X, steps_per_epoch=steps, epochs=120, callbacks=[evaluator])
