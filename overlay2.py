import cv2
import numpy as np

s_img = cv2.imread("smaller_image.png")
l_img = cv2.imread("larger_image.jpg")
x_offset=y_offset=50
l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

cv2.imshow("imagen", l_img)
cv2.waitKey(0)
