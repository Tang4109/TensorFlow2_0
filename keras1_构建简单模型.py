'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/22 17:50
@Author  : Zhangyunjia
@FileName: keras1_构建简单模型.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''
# 1.导入tf.keras
import tensorflow as tf
from tensorflow.keras import layers

print(tf.__version__)
print(tf.keras.__version__)
# 2.构建简单模型
# 模型堆叠
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# 网络配置
'''
layers.Dense(32, activation=tf.sigmoid)
layers.Dense(32, kernel_initializer='orthogonal')
layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal)
layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))
layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(0.01))
'''

# 3.训练和评估
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# 通过numpy输入数据
import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))
val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))
# model.fit(train_x, train_y, batch_size=100, validation_data=(val_x, val_y))

# 通过tf.data输入数据
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(32)
dataset = dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()
model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=val_dataset, validation_steps=3)

# 评估与测评
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
# model.evaluate(test_x,test_y, batch_size =32)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)
# 预测
result = model.predict(test_x, batch_size=32)
print(result)

