#!/usr/bin/env python
# coding: utf-8
# # 数据准备
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from time import time
import numpy as np
import matplotlib.pyplot as plt
def train():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # # 建立模型
    def layer(output_dim, input_dim, inputs, activation=None):
        W = tf.Variable(tf.random_normal([input_dim, output_dim]))
        b = tf.Variable(tf.random_normal([1, output_dim]))
        XWb = tf.matmul(inputs, W) + b
        if activation is None:
            outputs = XWb
        else:
            outputs = activation(XWb)
        return outputs

    # 建立输入层 x

    x = tf.placeholder("float", [None, 784],name='myInput')

    # 建立隐藏层h1

    h1 = layer(output_dim=256, input_dim=784,
               inputs=x, activation=tf.nn.relu)

    # 建立输出层

    y_predict = layer(output_dim=10, input_dim=256,
                      inputs=h1, activation=None)
    tf.identity(y_predict,name='myOutput')
    # # 定义训练方式
    # 建立训练数据label真实值 placeholder

    y_label = tf.placeholder("float", [None, 10])

    # 定义loss function

    loss_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits
        (logits=y_predict,
         labels=y_label))

    # 选择optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

    # # 定义评估模型的准确率
    # 计算每一项数据是否正确预测

    correct_prediction = tf.equal(tf.argmax(y_label, 1),
                                  tf.argmax(y_predict, 1))
    # 将计算预测正确结果，加总平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # # 开始训练

    trainEpochs = 15
    batchSize = 100
    totalBatchs = int(mnist.train.num_examples / batchSize)
    epoch_list = []
    loss_list = []
    accuracy_list = []


    startTime = time()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(trainEpochs):
        for i in range(totalBatchs):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})

        loss, acc = sess.run([loss_function, accuracy],
                             feed_dict={x: mnist.validation.images,
                                        y_label: mnist.validation.labels})

        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print("Train Epoch:", '%02d' % (epoch + 1), "Loss=", "{:.9f}".format(loss), " Accuracy=", acc)

    duration = time() - startTime
    print("Train Finished takes:", duration)

    #get_ipython().run_line_magic('matplotlib', 'inline')

    # fig = plt.gcf()
    # fig.set_size_inches(4, 2)
    # plt.plot(epoch_list, loss_list, label='loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['loss'], loc='upper left')
    #
    # plt.plot(epoch_list, accuracy_list, label="accuracy")
    # fig = plt.gcf()
    # fig.set_size_inches(4, 2)
    # plt.ylim(0.8, 1)
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.show()

    # # 评估模型的准确率

    print("Accuracy:", sess.run(accuracy,
                                feed_dict={x: mnist.test.images,
                                           y_label: mnist.test.labels}))

    #保存模型
    tf.saved_model.simple_save(sess,
                               './model',
                               inputs={'myInput':x},
                               outputs={'myOutput':y_predict})


    sess.close()
