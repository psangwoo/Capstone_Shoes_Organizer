import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from organize import organize
from pair.find import match_shoes


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
# Grab path to current working directory
CWD_PATH = os.getcwd()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = 'shoeDetection/frozen_inference_graph.pb'

# Path to label map file
PATH_TO_LABELS = 'shoeDetection/label_map.pbtxt'

# Path to image
PATH_TO_IMAGE_FOLDER = 'shoeDetection/imgs'

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

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


filename = os.path.join(PATH_TO_IMAGE_FOLDER, "Shoes.jpg")


print(filename)
image = cv2.imread(filename)
put_image = image.copy()
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
vis_util.visualize_boxes_and_labels_on_image_array(
    put_image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=5,
    min_score_thresh=0.50)

# All the results have been drawn on image. Now display the image.
max_size = [0, 0]
width, height, dim = image.shape

shoes = []

for i, b in enumerate(scores[0]):
    if b>0.5:
        y, x, y1, x1 = boxes[0][i]
        x*=height
        x1*=height
        y*=width
        y1*=width
        x = int(x)
        y = int(y)
        x1 = int(x1)
        y1 = int(y1)
        
        if max_size[0]<x1-x+40:
            max_size[0] = x1-x+40
        if max_size[1]<y1-y+40:
            max_size[1] = y1-y+40
        shoes.append((x-20, y-20, x1+20, y1+20))

filename = os.path.join(PATH_TO_IMAGE_FOLDER, "Background.jpg")
background = cv2.imread(filename)
background = cv2.resize(background, dsize=(image.shape[0], image.shape[1]), interpolation=cv2.INTER_AREA)
cv2.imshow("found", put_image)

#pairs = [[5, 0], [1, 2], [4, 3]]

pairs = match_shoes(image, shoes)
print(pairs)

destinations = {}

max_fit = 0
w = 0
while w<height:
    max_fit+=2
    w+=max_size[0]*2
max_fit-=2

sort_shoe = sorted(shoes, key = lambda item: ((item[0]**2 + item[1]**2)), reverse=False)
print(sort_shoe)
print(shoes)

num=0
for j, shoe in enumerate(sort_shoe):
    if shoe==-1:
        continue
    i = shoes.index(shoe)
    destinations[i] = (max_size[0]*((2*num)%max_fit), max_size[1]*((2*num)//max_fit))
    for p in pairs:
        if p[0] == i:
            destinations[p[1]] = (max_size[0]*((2*num+1)%max_fit), max_size[1]*((2*num+1)//max_fit))
            sort_shoe[sort_shoe.index(shoes[p[1]])] = -1
            sort_shoe[j] = -1
        elif p[1] == i:
            destinations[p[0]] = (max_size[0]*((2*num+1)%max_fit), max_size[1]*((2*num+1)//max_fit))
            sort_shoe[sort_shoe.index(shoes[p[0]])] = -1
            sort_shoe[j] = -1
    num+=1
    print(destinations, i)

final_destinations = sorted(destinations.items(), key = lambda item: (item[1][1], item[1][0]), reverse=False)
#final_destinations = sorted(destinations.items(), key = lambda item: (item[1][1], (shoes[item[0]][0]-item[1][0])**2 + (shoes[item[0]][1]-item[1][1])**2), reverse=False)
#final_destinations = sorted(destinations.items(), key = lambda item: max(shoes[item[0]][1]-item[1][1], shoes[item[0]][0]-item[1][0]), reverse=False)
#final_destinations = sorted(destinations.items(), key = lambda item: max(shoes[item[0]][1], shoes[item[0]][0]), reverse=False)
#print(final_destinations)
i=0
while i<len(final_destinations):
    print(final_destinations[i:])
    #print(shoes)
    shoe, destination = final_destinations[i]
    X, Y = destination
    X1, Y1 = X+max_size[0], Y+max_size[1]
    t = ()
    
    for j, (x, y, x1, y1) in enumerate(shoes):
        if j==shoe:
            continue
        if X1>x and X<x1 and Y1>y and Y<y1:
            print("intersect")
            for k, (num, des) in enumerate(final_destinations):
                if j==num:
                    t= (num, des)
                    del final_destinations[k]
                    final_destinations.insert(i, t)
                    break
            if t!=():
                break
    if t!=():
        continue
            
    image = organize(image, background, shoes[shoe], destination)
    shoes[shoe] = (X, Y, X1, Y1)
    cv2.imshow(str(i), image)
    i+=1
    cv2.imwrite("organized_{}.jpg".format(str(i)), image)
    cv2.waitKey()


cv2.destroyAllWindows()
