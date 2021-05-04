import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import warnings
import os
import cv2
import sys
import time
import json
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import json_utils
from object_detection.protos import eval_pb2
from collections import defaultdict
from PIL import Image

sys.path.append("..")

def detection():
    image_np = cv2.imread(PATH_TO_IMAGE)
    input_tensor = tf.convert_to_tensor(image_np)
    image_copy = input_tensor[tf.newaxis, ...]
    detections = detect_fn(image_copy)
    # print(detections)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    # boxes = np.squeeze(detections['detection_boxes'])
    # scores = np.squeeze(detections['detection_scores'])
    # classes = np.squeeze(detections['detection_classes']).astype(np.int64)
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.85,
        agnostic_mode=False
    )
 

    coordinates = viz_utils.return_coordinates(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

    
    for coordinate in coordinates:
                print(coordinate)
                #ymin,ymax,xmin,xmax
                (y1, y2, x1, x2, accuracy, classification) = coordinate
                # if(len(classification)>1):
                #     print()

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
    
    output = image_np_with_detections.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120)
    circle_data = []
    print(circles)
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
        
    if circles is None:
        circle_data = [int(484),int(484),int(486)]
    #Image.fromarray(output).show()    
    with open(os.path.join(CWD_PATH,"json_hand/"+"script_"+IMAGE_NAME.split(".")[0]+".json"), "w",encoding='utf-8') as f:
        data = {
            'coordinate' : new_data,
            'circle' : circle_data
        }
        json.dump(data, f,ensure_ascii=False)
        f.write('\n') 
        
    Image.fromarray(output).show()
    #cv2.imwrite(PATH_TO_RESULT,output)
    print("save success!")

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()


PATH_TO_MODEL_DIR ='training_num\exported_model\\'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "\saved_model"
# Grab path to current working directory
CWD_PATH = os.getcwd()
#loop test all image
IMAGETEST_FOLDER = 'image_test'
CDT_PATH = 'CDT_rewrite'
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
CDT_Rewrite = 'result\CDT_rewrite\\'
ALL_HANDS = 'result\\allhands\\'
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_hand','label_map.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 2
###
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#folder all image test
folder = os.path.join(CWD_PATH,IMAGETEST_FOLDER,CDT_PATH)
id_folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
    
for i in range(len(id_folder)):
    print(id_folder[i])
    ID = str(id_folder[i])
    #ID = 'klqqlptyt'
    #IMAGE_NAME = ID+'_num.jpg'
    IMAGE_NAME = ID+'_hand.jpg'
    IMAGE_FOLDER = 'image_test\\CDT_rewrite\\'
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_FOLDER,ID,IMAGE_NAME)
    print(PATH_TO_IMAGE)
    RESULT_FOLDER = 'result\CDT_rewrite\\'+ ID
    CREATE = os.path.join(CWD_PATH,CDT_Rewrite,ID)
    PATH_TO_RESULT = os.path.join(CWD_PATH,'result\\CDT_rewrite',ID,IMAGE_NAME)
    print(PATH_TO_RESULT)
    try: 
        os.mkdir(CREATE)
    except OSError as error: 
        print(error)  
    detection()
    

