#선형 회기(Linear Regression)을 학습하기 위해 작성한 코드 입니다.

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

xData = [1,2,3,4,5,6,7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]

W = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X +b
cost = tf.reduce_mean(tf.square(H - Y))
a = tf.Variable(0.01)

optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)

for i in range(50001):
    sess.run(train, feed_dict = {X : xData, Y:yData})
    if i%500 == 0:
        print(i, sess.run(cost , feed_dict={X: xData, Y:yData}), sess.run(W), sess.run(b))
print(sess.run(H, feed_dict={X :[1]}))
