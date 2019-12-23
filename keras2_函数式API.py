'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/22 21:56
@Author  : Zhangyunjia
@FileName: keras2_函数式API.py
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

loss,accuracy = model.evaluate(test_x,test_y)
print(loss,accuracy)
# 预测
result = model.predict(test_x)
print(result)



