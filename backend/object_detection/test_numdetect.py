# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
#from utils import eval_util as eval_utils
from object_detection.utils import json_utils
from object_detection.protos import eval_pb2
from collections import defaultdict
def detection():

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

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
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
                #print(coordinate)
                #ymin,ymax,xmin,xmax
                (y1, y2, x1, x2, accuracy, classification) = coordinate
                #print(accuracy)

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
        

    with open(os.path.join(CWD_PATH,"json_num/CDT_rewrite/"+"script_"+IMAGE_NAME.split(".")[0]+".json"), "w",encoding='utf-8') as f:
        data = {
            'coordinate' : new_data,
            'circle' : circle_data
        }
        json.dump(data, f,ensure_ascii=False)
        f.write('\n')
        
    Image.fromarray(output).show()
    cv2.imwrite(PATH_TO_RESULT,output)

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()

# Grab path to current working directory
CWD_PATH = os.getcwd()

#loop test all image
IMAGETEST_FOLDER = 'image_test'
CDT_PATH = 'CDT_rewrite'
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
CDT_Rewrite = 'result\CDT_rewrite\\new'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph_number.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap_number.pbtxt')


# Number of classes the object detector can identify
NUM_CLASSES = 12

folder = os.path.join(CWD_PATH,IMAGETEST_FOLDER,CDT_PATH)
id_folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

for i in range(len(id_folder)):
    #print(id_folder[i])
    #ID = str(id_folder[i])
    ID = 'test11'
    #IMAGE_NAME = ID+'_num.jpg'
    IMAGE_NAME = ID+'.png'
    IMAGE_FOLDER = 'image_test\\new\\'
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_FOLDER,IMAGE_NAME)
    print(PATH_TO_IMAGE)
    RESULT_FOLDER = 'result\CDT_rewrite\\new\\'+ ID
    CREATE = os.path.join(CWD_PATH,CDT_Rewrite,ID)
    PATH_TO_RESULT = os.path.join(CWD_PATH,'result\\new',IMAGE_NAME)
    try: 
        os.mkdir(CREATE)
    except OSError as error: 
        print(error)  
    detection()
    break


# Name of the directory containing the object detection module we're using
# MODEL_NAME = 'inference_graph'
# IMAGE_NAME = ID+'_num.jpg'
# DETECT_FOLDER = 'detectcircle'
# CDT_Rewrite = 'result\CDT_rewrite\\'
# IMAGE_FOLDER = 'image_test\CDT_rewrite\\'+ ID
# RESULT_FOLDER = 'result\CDT_rewrite\\'+ ID
#os.mkdir(CDT_Rewrite, ID)

    





#detection()
