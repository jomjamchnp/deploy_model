from flask import Flask
# from model.hand import test
import os
import sys
sys.path.append('..')
import numpy as np
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db
# from firebase import firebase
from backend.object_detection.db_num import num_detection
from backend.object_detection.score_num import score_num
import pyrebase
import urllib.request

config = {
  "apiKey": "AIzaSyDeetZdGCCL3UJAVkT-um9Z8NxHM9zGnAg",
  "authDomain": "psychopaint-app.firebaseapp.com",
  "databaseURL": "https://psychopaint-app.firebaseio.com",
  "storageBucket": "psychopaint-app.appspot.com",
  "serviceAccount": "./psychopaint-app-firebase-adminsdk-jcnly-e09b3ce809.json"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
app = Flask(__name__)

# img = db.child("CDT/_09pgfzb45/drawing_info/number/url").get()

@app.route('/test',methods=['GET'])
def main():
    name = '_vgh42ixnj'
    CWD_PATH = os.getcwd()
    # KEY = os.path.join(CWD_PATH,"psychopaint-app-firebase-adminsdk-jcnly-c9ee2ded1d.json")
    # cred = credentials.Certificate(KEY)
    # firebase = firebase.FirebaseApplication('https://psychopaint-app.firebaseio.com', None)
    # url_num = firebase.get('/CDT/'+name+'/drawing_info/number/url','')
    # url_total = firebase.get('/CDT/'+name+'/drawing_info/hand/url','')
    pic_num = db.child("CDT/"+name+"/drawing_info/number/url").get()
    url = pic_num.val()
    resp = urllib.request.urlopen(url)
    image_num = np.asarray(bytearray(resp.read()), dtype="uint8")
    num_detection(name,image_num)
    score_1,score_2,score_3 = score_num(name,image_num)
    print(score_1,score_2,score_3)
    # pic_total = db.child("CDT/"+name+"/drawing_info/hand/url").get()
    # return image_pro.diff_img(name)
    return score_1,score_2,score_3

if __name__ == '__main__':
    app.run()