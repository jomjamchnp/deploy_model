from flask import Flask
from model.hand import test
import os
import image_pro
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase import firebase
import urllib.request
app = Flask(__name__)


@app.route('/')
def hello():
    name = '_vgh42ixnj'
    CWD_PATH = os.getcwd()
    KEY = os.path.join(CWD_PATH,"psychopaint-app-firebase-adminsdk-jcnly-c9ee2ded1d.json")
    cred = credentials.Certificate(KEY)
    firebase = firebase.FirebaseApplication('https://psychopaint-app.firebaseio.com', None)
    url_num = firebase.get('/CDT/'+name+'/drawing_info/number/url','')
    url_total = firebase.get('/CDT/'+name+'/drawing_info/hand/url','')
    
    return image_pro.diff_img(name)

if __name__ == '__main__':
    app.run()