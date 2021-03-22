import sys
sys.path.append('..')
from backend.object_detection.db_num_1 import num_detection
from backend.object_detection.score_num_1 import score_num
from backend.object_detection.db_hand_1 import hand_detection
from backend.object_detection.score_hand_1 import score_hand
from backend.object_detection.image_pro import diff_img

import cv2
import os
import pyrebase
import numpy as np
import urllib.request
from PIL import Image

config = {
  "apiKey": "AIzaSyDeetZdGCCL3UJAVkT-um9Z8NxHM9zGnAg",
  "authDomain": "psychopaint-app.firebaseapp.com",
  "databaseURL": "https://psychopaint-app.firebaseio.com",
  "storageBucket": "psychopaint-app.appspot.com",
  "serviceAccount": "./psychopaint-app-firebase-adminsdk-jcnly-e09b3ce809.json"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()

name = "_a0qj9waa8"
# get image from database 
pic_num = db.child("CDT/"+name+"/drawing_info/number/url").get()
pic_hand = db.child("CDT/"+name+"/drawing_info/hand/url").get()
# url image number and total
url_num = pic_num.val()
url_hand = pic_hand.val()
# convert url to image array
resp_num = urllib.request.urlopen(url_num)
resp_hand = urllib.request.urlopen(url_hand)
image_num = np.asarray(bytearray(resp_num.read()), dtype="uint8")
image_hand = np.asarray(bytearray(resp_hand.read()), dtype="uint8")
img_test_number = cv2.imdecode(image_num, cv2.IMREAD_COLOR)
image_2 = cv2.imdecode(image_hand, cv2.IMREAD_COLOR)
# image processing for different image to image hand
img_test_hand = diff_img(img_test_number,image_2)
# push image to model and calculate score
num_detection(name,img_test_number)
hand_detection(name,img_test_hand)
score_1,score_2,score_3 = score_num(name,img_test_number)
score_4,score_5 = score_hand(name,img_test_hand)
print("score 1,2,3: ",score_1,score_2,score_3)
print("score 4,5: ",score_4,score_5)
print("total :",score_1+score_2+score_3+score_4+score_5)

