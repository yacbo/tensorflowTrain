# -*- coding: utf-8 -*-

import cv2

img = cv2.imread("aaa.jpg")
# print(img);
# 在窗口中显示图像
cv2.imshow("Image", img)
# 如果不添waitKey ，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过
cv2.waitKey(10000)
cv2.destroyAllWindows()

