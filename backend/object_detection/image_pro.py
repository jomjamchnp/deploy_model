import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase import firebase
import urllib.request

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image



def diff_img(name):
	ID_NAME = name
	FILE = '.jpg'
	IMAGE_FOLDER = 'image_test'
	CWD_PATH = os.getcwd()
	KEY = os.path.join(CWD_PATH,"psychopaint-app-firebase-adminsdk-jcnly-c9ee2ded1d.json")
		####DATABASE
	cred = credentials.Certificate(KEY)
		# firebase_admin.initialize_app(cred, {
		# 	'databaseURL': 'https://psychopaint-app.firebaseio.com'
		# })
	firebase = firebase.FirebaseApplication('https://psychopaint-app.firebaseio.com', None)
	url_num = firebase.get('/CDT/'+ID_NAME+'/drawing_info/number/url','')
	url_total = firebase.get('/CDT/'+ID_NAME+'/drawing_info/hand/url','')
	#Path JSON firebase


	img_total = url_to_image(url_total)
	img_num = url_to_image(url_num)

	img_hands = cv2.absdiff(img_total, img_num)
	img_last = cv2.bitwise_not(img_hands)

	cv2.imshow('img_hands',img_last)
	cv2.imwrite(os.path.join(IMAGE_FOLDER,ID_NAME+'_hands.jpg'),img_last)
	cv2.imwrite(os.path.join(IMAGE_FOLDER,ID_NAME+'_num.jpg'),img_num)
	print(os.path.join(IMAGE_FOLDER,ID_NAME))
	cv2.waitKey(0)
# diff_img()
