import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
from PIL import ImageFilter

ID_NAME = '_h9ezb07yl'
FILE = '.jpg'
IMAGE_FOLDER = 'image_test'

CWD_PATH = os.getcwd()

PATH_TO_IMAGE_FRAME = os.path.join(IMAGE_FOLDER,ID_NAME+'frame'+FILE)
PATH_TO_IMAGE_NUM = os.path.join(IMAGE_FOLDER,ID_NAME+'_num'+FILE)

img1 = cv2.imread(PATH_TO_IMAGE_FRAME,cv2.IMREAD_COLOR)
img2 = cv2.imread(PATH_TO_IMAGE_NUM,cv2.IMREAD_COLOR)

img3 = cv2.addWeighted(img1,0.5,img2,0.5,0)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.erode(img3,kernel,iterations = 1)

cv2.imshow('img3',dilation)
#cv2.imwrite(os.path.join(IMAGE_FOLDER,ID_NAME+'.jpg'),img3)


cv2.waitKey(0)


