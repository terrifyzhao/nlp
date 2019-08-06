import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from text_matching.model import model
from text_matching.train import extract_entity

model = model()
model.load_weights('../matching_model.weights')


def predict(text1, text2):
    res, prob = extract_entity(text1, text2, model)
    return res, prob


if __name__ == '__main__':
    while 1:
        text1 = input('content:')
        text2 = input('content:')
        r, prob = predict(text1, text2)
        print(r, '----', str(prob))
