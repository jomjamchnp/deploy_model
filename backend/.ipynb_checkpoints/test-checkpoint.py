import sys
sys.path.append('..')
from backend.object_detection.db_num import num_detection
from backend.object_detection.score_num import score_num
import cv2
import os

CWD_PATH = os.getcwd()
PATH_TO_IMAGE = os.path.join(CWD_PATH,'/testP66.JPG')
output = cv2.imread('./object_detection/image_test/num1.jpg')
# output = cv2.imread('./object_detection/image_test/num1.jpg')
name = "test"
num_detection(name,output)

