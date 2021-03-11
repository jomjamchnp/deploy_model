
######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import urllib
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
from PIL import Image
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore
from firebase import firebase
import urllib.request
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
#from utils import eval_util as eval_utils
from object_detection.utils import json_utils
from object_detection.protos import eval_pb2

# Grab path to current working directory
CWD_PATH = os.getcwd()
# #Path JSON firebase
# KEY = os.path.join(CWD_PATH,"psychopaint-app-firebase-adminsdk-jcnly-c9ee2ded1d.json")

# ####DATABASE
# # Fetch the service account key JSON file contents
# cred = credentials.Certificate(KEY)
# # Initialize the app with a custom auth variable, limiting the server's access
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://psychopaint-app.firebaseio.com'
# })

# firebase = firebase.FirebaseApplication('https://psychopaint-app.firebaseio.com', None)
# result = firebase.get('/CDT/'+ID_NAME+'/drawing_info/number/url','')


def url_to_image(url):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def num_detection(name):
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    ID_NAME = name
    FILE = '_num.jpg'
    IMAGE_FOLDER = 'image_test'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph_number.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap_number.pbtxt')
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_FOLDER,ID_NAME+FILE)

    #Path JSON firebase
    # Number of classes the object detector can identify
    NUM_CLASSES = 12
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image = cv2.imread(PATH_TO_IMAGE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.7)

    coordinates = vis_util.return_coordinates(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.7)

    for coordinate in coordinates:
                print(coordinate)
                #ymin,ymax,xmin,xmax
                (y1, y2, x1, x2, accuracy, classification) = coordinate


    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100)

    circle_data = []

    # ensure at least some circles were found
    if circles is not None:
        #convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # line vertical
            start_ver, stop_ver = (x-r, y), (x+r, y)
            # line horizontal
            start_hor, stop_hor = (x, y-r), (x, y+r)
            #draw a line cv2.line(img, Point pt1, Point pt2, color[,thickness[,lineType[,shift]]])
            cv2.line(output,start_ver,stop_ver, (0,0,0), (3))
            cv2.line(output,start_hor,stop_hor, (0,0,0), (3))
            circle_data = [int(x),int(y),int(r)]
        

    with open(os.path.join(CWD_PATH,"json_num/"+"script"+ID_NAME.split(".")[0]+".json"), "w",encoding='utf-8') as f:
        data = {
            'coordinate' : coordinates,
            'circle' : circle_data
        }
        json.dump(data, f,ensure_ascii=False)
        f.write('\n')

    #cv2.imshow('Object detector', output)
    Image.fromarray(output).show()

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()
num_detection('_vgh42ixnj')