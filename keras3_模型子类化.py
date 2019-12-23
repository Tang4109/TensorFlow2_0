'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/23 8:38
@Author  : Zhangyunjia
@FileName: keras3_模型子类化.py
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
y = np.random.random((2000, 10))
train_x, test_x, train_y, test_y = train_test_split(X, y)


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32, activation='relu')
        self.layer2 = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=16, epochs=5)
print('////////////////////////////////////')
loss, accuracy = model.evaluate(test_x, test_y)
print(loss,accuracy)
# 预测
result = model.predict(test_x)
print(result)
