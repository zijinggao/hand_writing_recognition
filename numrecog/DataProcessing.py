import base64
import cv2
from PIL import Image
from array import *
import tensorflow as tf
import numpy as np


def transformData(img):
    img1=img.replace('data:image/png;base64,','')
    imgData = base64.b64decode(img1)
    file = open('test.png', 'wb')
    file.write(imgData)
    file.close()

def cvgray(filename):
    #img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img=cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename, gray)

def creatData(filename):
    imageData = array('B')
    Im = Image.open(filename)
    pixel = Im.load()
    width, height = Im.size
    for x in range(0, width):
        for y in range(0, height):
            imageData.append(pixel[y, x])
    Data = [0]*784
    for i in range(len(imageData)):
        Data[i]=round((255-imageData[i])/255,7)
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
        #print(np.argmax(scores, 1))
        # plt.imshow(np.reshape(xx, (28, 28)),
        #            cmap='binary')
        # plt.show()
        return np.argmax(scores, 1)[0]

def recognize(img):
    transformData(img)
    cvgray('test.png')
    Data=creatData('test.png')
    ans = usemodel(Data)
    return ans