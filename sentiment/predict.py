import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from sentiment.sentiment_model import sentiment_model
from sentiment.sentiment_train import extract_entity

model = sentiment_model()
model.load_weights('../output/sentiment_model.weights')


def predict(content):
    res, prob = extract_entity(content, model)
    if int(res) == 0:
        return '中性', prob
    elif int(res) == 1:
        return '正面', prob
    elif int(res) == 2:
        return '负面', prob
    elif int(res) == 3:
        return '未提及', prob
    return '其它', prob


if __name__ == '__main__':
    while 1:
        content = input('content:')
        r, prob = predict(content)
        print(r, '----', str(prob))
