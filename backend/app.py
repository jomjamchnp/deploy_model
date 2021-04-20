from flask import Flask, request, Response, jsonify
# from model.hand import test
import os
import sys
sys.path.append('..')
import numpy as np
import cv2
# from backend.object_detection.db_num_1 import num_detection
# from backend.object_detection.score_num_1 import score_num
# from backend.object_detection.db_hand_1 import hand_detection
# from backend.object_detection.score_hand_1 import score_hand
# from backend.object_detection.image_pro import diff_img
from object_detection.db_num_1 import num_detection
from object_detection.score_num_1 import score_num
from object_detection.db_hand_1 import hand_detection
from object_detection.score_hand_1 import score_hand
from object_detection.image_pro import diff_img
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

@app.route('/test/scoreCDT',methods=['POST'])
def main():
  if request.method == 'POST':
      url1 = request.form.get('url1')
      url2 = request.form.get('url2')
  print(url1)
  print(url2)
  # name = '_vgh42ixnj'
  # name = "_09pgfzb45"
  name = "Test"
  # get image from database 
  # pic_num = db.child("CDT/"+name+"/drawing_info/number/url").get()
  # pic_hand = db.child("CDT/"+name+"/drawing_info/hand/url").get()

  # url image number and total
  # url_num = url1.val()
  # url_hand = url2.val()
  # convert url to image array
  resp_num = urllib.request.urlopen(url1)
  resp_total = urllib.request.urlopen(url2)
  image_num = np.asarray(bytearray(resp_num.read()), dtype="uint8")
  image_total = np.asarray(bytearray(resp_total.read()), dtype="uint8")
  img_test_number = cv2.imdecode(image_num, cv2.IMREAD_COLOR)
  image_2 = cv2.imdecode(image_total, cv2.IMREAD_COLOR)
  # image processing for different image to image hand
  img_test_hand = diff_img(img_test_number,image_2)
  # push image to model and calculate score
  num_detection(name,img_test_number)
  hand_detection(name,img_test_hand)
  score_1,score_2,score_3 = score_num(name,img_test_number)
  score_4,score_5 = score_hand(name,img_test_hand)
  print("score 1,2,3: ",score_1,score_2,score_3)
  print("score 4,5: ",score_4,score_5)
  total_score = score_1+score_2+score_3+score_4+score_5
  print("total :",total_score)
  response = {
      "score_1": score_1,
      "score_2": score_2,
      "score_3": score_3,
      "score_4": score_4,
      "score_5": score_5,
      "total_score": total_score
  }
  # return "total : "+str(total_score)
  return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT',8000))
    app.run(debug=True, host='0.0.0.0', port= port)