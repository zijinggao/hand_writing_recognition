import base64
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np

def convert_png_uri_to_img(uri):
    encoded_data = uri.split(',')[1]
    decoded_bytes = base64.b64decode(encoded_data)
    im = Image.open(BytesIO(decoded_bytes))
    _, _, _, img = im.split()
    return np.array(img)

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
    Data = im.flatten().tolist()
    y=usemodel(Data)
    return int(y)