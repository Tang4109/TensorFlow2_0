'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/23 9:05
@Author  : Zhangyunjia
@FileName: keras4_自定义层.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''
# 1.导入tf.keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# 通过numpy输入数据
import numpy as np

X = np.random.random((2000, 72))
y = np.random.random((2000, 5))
train_x, test_x, train_y, test_y = train_test_split(X, y)


class MyLayer(layers.Layer):
    # *args用来将参数打包成tuple给函数体调用
    # **kwargs 打包关键字参数成dict给函数体调用
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel1', shape=shape, initializer='uniform', trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model = tf.keras.Sequential([MyLayer(5), layers.Activation('softmax')])
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=5)
loss, accuracy = model.evaluate(test_x, test_y)
print(loss,accuracy)
# 预测
result = model.predict(test_x)
print(result)
print(model.get_config())

