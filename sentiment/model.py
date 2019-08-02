from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint
import keras
import tensorflow as tf

embedding_size = 768
config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'

global graph
graph = tf.get_default_graph()

g_nums = 2


def sentiment_model(learning_rate=1e-5):
    with graph.as_default():
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        x_in = Input(shape=(None,))  # 待识别句子输入
        x_in2 = Input(shape=(None,))  # 待识别句子输入

        x = bert_model([x_in, x_in2])
        x = Lambda(lambda x: x[:, 0])(x)
        # out = Dense(128, use_bias=False)(x)
        out = Dense(4, activation='softmax')(x)

        model = Model([x_in, x_in2], out)
        # model = keras.utils.multi_gpu_model(M, gpus=g_nums)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))
        model.summary()

        return model
