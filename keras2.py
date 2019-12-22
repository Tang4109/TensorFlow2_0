'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/22 21:56
@Author  : Zhangyunjia
@FileName: keras2.py
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
train_x, test_x,train_y, test_y = train_test_split(X, y)

# 4.构建高级模型
# 函数式API
input_x = tf.keras.Input(shape=(72,))
hidden1 = layers.Dense(32, activation='relu')(input_x)
hidden2 = layers.Dense(16, activation="relu")(hidden1)
pred = layers.Dense(10, activation="softmax")(hidden2)
model = tf.keras.Model(inputs=input_x, outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=32, epochs=5)
# 评估与测评

ratio = model.evaluate(test_x,test_y, batch_size =32)
print(ratio)
# 预测
result = model.predict(test_x, batch_size=32)
print(result)


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

# model = MyModel(num_classes=10)
# model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
#               loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
# model.fit(train_x, train_y, batch_size=16, epochs=5)
# model.evaluate(test_data, steps=30)
