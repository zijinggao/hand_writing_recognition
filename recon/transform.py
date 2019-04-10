import struct
import numpy as np

##用于读取mnist格式的数据集生成可导入模型
def read_images(enter):
    data = enter.read()
    head = struct.unpack_from('>IIII', data, 0)
    offset = struct.calcsize('>IIII')
    number = head[1]
    width = head[2]
    height = head[3]
    bits = number * width * height
    bitsString = '>' + str(bits) + 'B'
    images = struct.unpack_from(bitsString, data, offset)
    images = np.reshape(images, [number, width *height])
    return number,images


enter = open('1.idx3-ubyte','rb')
num,img=read_images(enter)
test =[0]*784
for i in range(len(img[0])):
    test[i]=round((255-img[0][i])/255,7)


