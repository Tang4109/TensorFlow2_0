'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/25 10:41
@Author  : Zhangyunjia
@FileName: 低阶API基础.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

import tensorflow as tf

# 1.常量声明tf.constant
a = tf.constant(7)
print(a)

# 2.变量声明tf.Variable
print('////////////////////////////')
a1 = 7
a2 = tf.Variable(7)
a3 = tf.Variable([0, 1, 2])
print(a1)
print(a2)
print(a3)
# 3.多阶Tensor的形状变换tf.reshape
print('///////////////////////////')
a4 = tf.Variable([[0, 1, 2], [3, 4, 5]])
print(a4.shape)
a5 = tf.reshape(a4, [3, 2])
print(a5.shape)
print(a5)
# 4.对Tensor求平均值tf.math.reduce_mean
print('////////////////////////////')
a = tf.constant([1, 2., 3, 4, 5, 6, 7.])
print(a.dtype)
print(tf.math.reduce_mean(a))
b = tf.constant([[1, 2, 1], [5, 2, 10]])
print(b.dtype)
print(tf.math.reduce_mean(b))

# 5.随机生成一个Tensor,其值符合正态分布tf.random.normal
print('//////////////////////////////')
a = tf.random.normal(shape=[2, 3], mean=2, stddev=0.2, dtype=tf.float32, seed=0)
print(a.numpy())

# 6.随机生成一个Tensor,其值符合均匀分布tf.random.uniform
print("///////////////////////////////")
a = tf.random.uniform(shape=[2, 3], minval=1, maxval=10, seed=8, dtype=tf.int32)
print(a.numpy())
# 7.矩阵转置tf.transpose
print("//////////////////////////////////////////////")
x = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(x)
a = tf.transpose(x, perm=[0, 2, 1])
print(a)

# 8.返回一个数组内最大值对应的索引tf.math.argmax
print('/////////////////////////////////////')
a = tf.constant([1, 2, 3, 4, 5])
x = tf.math.argmax(a)
print(x)

# 9.在输入的Tensor中增加一个维度
print('/////////////////////')
a = tf.constant([[1], [2], [3]])
print(a)
b = tf.expand_dims(a, 0)
print(b)

# 10.将多个Tensor在同一个维度上进行连接
print('////////////////////////////////')
a1 = tf.constant([[2, 3, 4], [4, 5, 6], [2, 3, 4]])
a2 = tf.constant([[6, 8, 4], [7, 5, 9], [6, 5, 4]])

b = tf.concat([a1, a2], axis=1)
print(b)
# 11.数据类型转换tf.bitcast
print('///////////////////////////////')
a=tf.constant(32.0)
b=tf.bitcast(a,type=tf.int32)
print(a)
print(b)