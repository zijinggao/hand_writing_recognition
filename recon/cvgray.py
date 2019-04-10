import cv2
##图片灰度处理
img = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('2.png',img)
