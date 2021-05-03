import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db
# from firebase import firebase
import urllib.request

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def diff_img_getUrl(img_number,img_total):
	img_hands = cv2.absdiff(img_total, img_number)
	img_last = cv2.bitwise_not(img_hands)
	return img_last


def diff_img():
	# name,img_number,img_total
	#ID_NAME = name
	FILE = '.PNG'
	IMAGE_FOLDER = 'image_test'
	CWD_PATH = os.getcwd()
	CDT_PATH = 'CDT_rewrite'
	NEW_PATH = 'new'
	folder = os.path.join(CWD_PATH,IMAGE_FOLDER,NEW_PATH)
	id_folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
	
	for i in range(len(id_folder)):
		#print(id_folder[i])
		path = 'test11'
		# path = str(folder)+'\\'+str(id_folder[i])
		newpath = str(folder)+'\\'+str(path)
		img_total = cv2.imread(newpath+'_clock.png',0)
		print(img_total)
		kernel = np.ones((3,3),np.uint8)
		img_num = cv2.imread(newpath+'.png',0)
		img_hands = cv2.absdiff(img_total, img_num)
		opening = cv2.morphologyEx(img_hands, cv2.MORPH_OPEN, kernel)
		img_last = cv2.bitwise_not(opening)
		print(str(folder))
		#Image.fromarray(img_last).show()
		cv2.imwrite(str(folder)+'\\dplit\\'+str(path)+'_hands.jpg',img_last)
		cv2.imwrite(str(folder)+'\\dplit\\'+str(path)+'_num.jpg',img_num)
		break
	#print(os.path.join(CWD_PATH,IMAGE_FOLDER,NEXT_PATH))
	# KEY = os.path.join(CWD_PATH,"psychopaint-app-firebase-adminsdk-jcnly-c9ee2ded1d.json")
	# 	####DATABASE
	# cred = credentials.Certificate(KEY)
	# 	# firebase_admin.initialize_app(cred, {
	# 	# 	'databaseURL': 'https://psychopaint-app.firebaseio.com'
	# 	# })
	# firebase = firebase.FirebaseApplication('https://psychopaint-app.firebaseio.com', None)
	# url_num = firebase.get('/CDT/'+ID_NAME+'/drawing_info/number/url','')
	# url_total = firebase.get('/CDT/'+ID_NAME+'/drawing_info/hand/url','')
	#Path JSON firebase


	# img_total = url_to_image(url_total)
	# img_num = url_to_image(url_num)

	# img_hands = cv2.absdiff(img_total, img_number)
	# img_last = cv2.bitwise_not(img_hands)
	#return img_last
	# cv2.imshow('img_hands',img_last)
	# cv2.imwrite(os.path.join(IMAGE_FOLDER,ID_NAME+'_hands.jpg'),img_last)
	# cv2.imwrite(os.path.join(IMAGE_FOLDER,ID_NAME+'_num.jpg'),img_num)
	# print(os.path.join(IMAGE_FOLDER,ID_NAME))
	#cv2.waitKey(0)
# diff_img()
