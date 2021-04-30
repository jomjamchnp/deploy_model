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

	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def sum_img():
	FILE = '.PNG'
	IMAGE_FOLDER = 'result\\CDT_rewrite'
	ORI = 'image_test\\CDT_rewrite'
	CWD_PATH = os.getcwd()
	RESULT_PATH = 'result\\all_score'
	result = os.path.join(CWD_PATH,RESULT_PATH)
	folder = os.path.join(CWD_PATH,IMAGE_FOLDER)
	ori_path = os.path.join(CWD_PATH,ORI)
	id_folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
	#print(id_folder)
	for i in range(0,len(id_folder)):
		print(i,id_folder[i])
		path = str(folder)+'\\'+str(id_folder[i])+'\\'+str(id_folder[i])
		#print(path)
		score_num = cv2.imread(path+'_scorenum.jpg')
		score_hand = cv2.imread(path+'_scorehand.jpg')
		h_img = cv2.vconcat([score_num, score_hand])
		img_ori = cv2.imread(ori_path+'\\'+str(id_folder[i])+'\clock.png')
		dim = (h_img.shape[0],h_img.shape[1])
		img_ori = cv2.resize(img_ori, dim, interpolation = cv2.INTER_AREA)
		total = cv2.hconcat([img_ori, h_img])
		#Image.fromarray(total).show()
		res = RESULT_PATH+str(id_folder[i])
		cv2.imwrite(RESULT_PATH+'\\'+str(id_folder[i])+'_all.jpg',total)
		

def diff_img():
	# name,img_number,img_total
	#ID_NAME = name
	FILE = '.PNG'
	IMAGE_FOLDER = 'image_test'
	CWD_PATH = os.getcwd()
	CDT_PATH = 'CDT_rewrite'
	NEW_PATH = 'new'
	folder = os.path.join(CWD_PATH,IMAGE_FOLDER,CDT_PATH)
	id_folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
	
	for i in range(len(id_folder)):
		print(id_folder[i])
		#id_folder[i]#path = 'test11'
		id_folder[i] = '0f6zqkidm'
		path = str(folder)+'\\'+str(id_folder[i])
		#newpath = str(folder)+'\\'+str(path)
		img_total = cv2.imread(path+'\clock.png',0)
		kernel = np.ones((3,3),np.uint8)
		img_num = cv2.imread(path+'\\'+'num.png',0)
		img_hands = cv2.absdiff(img_total, img_num)
		opening = cv2.morphologyEx(img_hands, cv2.MORPH_OPEN, kernel)
		img_last = cv2.bitwise_not(opening)
		print(str(folder))
		#Image.fromarray(img_last).show()
		cv2.imwrite((path+'\\'+str(id_folder[i]+'_hands.jpg')),img_last)
		cv2.imwrite((path+'\\'+str(id_folder[i]+'_num.jpg')),img_num)
		
	# KEY = os.path.join(CWD_PATH,"psychopaint-app-firebase-adminsdk-jcnly-c9ee2ded1d.json")
	# 	####DATABASE
	# cred = credentials.Certificate(KEY)
	# 	# firebase_admin.initialize_app(cred, {
	# 	# 	'databaseURL': 'https://psychopaint-app.firebaseio.com'
	# 	# })
	# firebase = firebase.FirebaseApplication('https://psychopaint-app.firebaseio.com', None)
	# url_num = firebase.get('/CDT/'+ID_NAME+'/drawing_info/number/url','')
	# url_total = firebase.get('/CDT/'+ID_NAME+'/drawing_info/hand/url','')

diff_img()

