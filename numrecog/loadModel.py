import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from time import time
import numpy as np

#mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
def hand_writing_reco():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess,['serve'],'./model')
        graph = tf.get_default_graph()

        #input=np.expand_dims(mnist.test.images[3],0)
        x = sess.graph.get_tensor_by_name('myInput:0')
        y = sess.graph.get_tensor_by_name('myOutput:0')
        batch_xs , batch_ys = mnist.test.next_batch(10)
        scores= sess.run(y,
                         feed_dict={x:batch_xs})
        print(np.argmax(scores,1))
#    for i in range(10):
 #       print('predict: %d, actual: %d' % (np.argmax(scores,1)[i],np.argmax(batch_ys,1))[i])
