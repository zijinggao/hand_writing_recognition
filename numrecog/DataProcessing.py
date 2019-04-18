import base64
import cv2
import tensorflow as tf
import numpy as np

def transformData(img):
    encoded_data = img.split(',')[1]
    imgData = base64.b64decode(encoded_data)
    file = open('test.png', 'wb')
    file.write(imgData)
    file.close()
def to8(filename):
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    (R, G, B, A) = cv2.split(im)
    imgData=[]
    Data = [0] * 784
    for i in A:
        imgData.extend(i)
    for i in range(len(imgData)):
        Data[i]=round((imgData[i])/255,7)
    return Data



def usemodel(Data):

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], './model')
        graph = tf.get_default_graph()
        x = sess.graph.get_tensor_by_name('myInput:0')
        y = sess.graph.get_tensor_by_name('myOutput:0')
        #输入网络的数据
        xx=[]
        xx.append(Data)
        scores = sess.run(y,
                          feed_dict={x: xx})
        return np.argmax(scores, 1)[0]

def recognize(img):
    transformData(img)
    Data=to8('test.png')
    y=usemodel(Data)
    return int(y)