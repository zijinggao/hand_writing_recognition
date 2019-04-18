import base64
import cv2
import tensorflow as tf
import numpy as np

def convert_png_uri_to_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    _, _, _, img = cv2.split(img)
    return img

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
    im = convert_png_uri_to_img(img) / 255
    #cv2.imwrite('test.bmp', im)
    #im = im / 255
    Data = im.flatten().tolist()
    y=usemodel(Data)
    return int(y)