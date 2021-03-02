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

CWD_PATH = os.getcwd()
#Path JSON firebase
KEY = os.path.join(CWD_PATH,"psychopaint-app-firebase-adminsdk-jcnly-c9ee2ded1d.json")

# ####DATABASE
# cred = credentials.Certificate(KEY)
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://psychopaint-app.firebaseio.com'
# })
# firebase = firebase.FirebaseApplication('https://psychopaint-app.firebaseio.com', None)
# result = firebase.get('/CDT/'+ID_NAME+'/drawing_info/number/url','')


def url_to_image(url):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

def hand_detection(name):
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    ID_NAME = name
    FILE = '_hands.jpg'
    DETECT_FOLDER = 'detectcircle'
    IMAGE_FOLDER = 'image_test'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_FOLDER,ID_NAME+FILE)
    NUM_CLASSES = 2

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

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load image using OpenCV and
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
        min_score_thresh=0.70)

    coordinates = vis_util.return_coordinates(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.70)

    for coordinate in coordinates:
            print(coordinate)
                #ymin,ymax,xmin,xmax
            (y1, y2, x1, x2, accuracy, classification) = coordinate

    with open(os.path.join(CWD_PATH,"json_hand/"+"script"+ID_NAME.split(".")[0]+".json"), "w",encoding='utf-8') as f:
        json.dump(coordinates, f,ensure_ascii=False, indent=4)
        f.write('\n')


    output = image.copy()
    cv2.imshow('Object detector', output)
    # Press any key to close the image
    cv2.waitKey(0)
    # Clean up
    cv2.destroyAllWindows()
 
