
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
import tensorflow.compat.v1 as tf
import sys
import json
from PIL import Image
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db
# from firebase_admin import firestore
# from firebase import firebase
import urllib.request
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from backend.object_detection.utils import label_map_util
from backend.object_detection.utils import visualization_utils as vis_util
#from utils import eval_util as eval_utils
from object_detection.utils import json_utils
from object_detection.protos import eval_pb2
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Grab path to current working directory
CWD_PATH = os.getcwd()
# tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


def url_to_image(url):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

# function for detect number 
def num_detection(name,image):
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    ID_NAME = name
    FILE = '_num.jpg'
    IMAGE_FOLDER = 'image_test'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = './object_detection/inference_graph/frozen_inference_graph_number.pb'
    PATH_TO_LABELS = './object_detection/training/labelmap_number.pbtxt'

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
            print("test print: ",od_graph_def.ParseFromString(serialized_graph))
            tf.import_graph_def(od_graph_def, name='')

        config = tf.compat.v1.ConfigProto(
            device_count = {'GPU': 0}
        )
        # sess = tf.compat.v1.Session(graph=detection_graph)
        sess = tf.Session(graph=detection_graph,config=config)

    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    print(image_tensor)
    print(detection_boxes)
    print(detection_scores)
    print(detection_classes)
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # convert image to rgb and shape in [None,w,h,c] 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    print(image_expanded.shape)

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
        min_score_thresh=0.3)

    coordinates = vis_util.return_coordinates(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.3)

    for coordinate in coordinates:
                print(coordinate)
                #ymin,ymax,xmin,xmax
                (y1, y2, x1, x2, accuracy, classification) = coordinate
   
    #get only high score
    new_data = []
    f = defaultdict(list)
    for i in range(0, len(coordinates)):
        num = coordinates[i][4]
        name = coordinates[i][5][0].split(":")
        f[name[0]].append(num)
    res =  list(zip(f, map(max, f.values())))
    list_index_hands = []
    for i in range(0,len(res)):
        idx = [x[4] for x in coordinates].index(res[i][1]) 
        list_index_hands.append(idx)
    new_data = []
    for i in list_index_hands:
        new_data.append(coordinates[i])

    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120)
    circle_data = []

    # ensure at least some circles were found
    if circles is None:
        circle_data = [int(484),int(484),int(486)]
    if circles is not None:
        #convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            cv2.circle(output,(x,y),r,(0,255,0),2)
            # line vertical
            start_ver, stop_ver = (x-r, y), (x+r, y)
            # line horizontal
            start_hor, stop_hor = (x, y-r), (x, y+r)
            #draw a line cv2.line(img, Point pt1, Point pt2, color[,thickness[,lineType[,shift]]])
            cv2.line(output,start_ver,stop_ver, (0,0,0), (3))
            cv2.line(output,start_hor,stop_hor, (0,0,0), (3))
            circle_data = [int(x),int(y),int(r)]
            print(circle_data)
            break
    
    
    cv2.circle(output,(circle_data[0],circle_data[1]),4,(255,0,0),2)
    # os.path.join(CWD_PATH,"json_num/"+"script"+ID_NAME.split(".")[0]+".json"
    with open("./object_detection/json_num/script_number.json", "w",encoding='utf-8') as f:
        data = {
            'coordinate' : new_data,
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
