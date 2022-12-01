# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================

import cv2 as cv
import numpy as np
import os
from PIL import Image
import gc
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

curr = os.getcwd()
curr = curr.replace('\\', '/')

#inputs
CONFIG      = './models/yolo/yoloV3.cfg'
WEIGHTS     = './models/yolo/yolov3.weights'
CLASSES     = './models/yolo/coco.txt'
INPUT_PATH  = curr + '/output/temp_count/images/0_raw/'
OUTPUT_PATH = curr + '/output/temp_count/images/1_cropped/'

classes = None
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def get_output_layers(net):
    layer_names = net.getLayerNames()
    temp = net.getUnconnectedOutLayers()
    # temp = [x[0] for x in temp]
    output_layers = [layer_names[i - 1] for i in temp]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    cv.rectangle(img, (int(x),int(y)), (int(x_plus_w),int(y_plus_h)), (255, 255, 255), 2)
    cv.imshow('image',img)
    cv.waitKey(0)  
    
def cropper(IMAGE_PATH, x, y, x_plus_w, y_plus_h, i, j):
    img = Image.open(IMAGE_PATH)
    width, height = img.size
    ans = img.crop((int(x), int(y), int(x_plus_w), int(y_plus_h)))
    xc = int((x + x_plus_w)/2/width*100)
    yc = int((y + y_plus_h)/2/height*100)    
    ans.save(OUTPUT_PATH + "{}_{}_{}.png".format(i,xc,yc))
    

def yolo():
    gc.collect()
    skip_classes = ['chair', 'person', 'umbrella', 'refrigerator', 'tv', 'bed', 'keyboard']
    imageList = os.listdir(INPUT_PATH)
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)
    detected = 0
    
    for i in range(len(imageList)):
        
        #specific image path
        IMAGE_PATH = INPUT_PATH + imageList[i]
        
        # read image
        image   = cv.imread(IMAGE_PATH)
        width   = image.shape[1]
        height  = image.shape[0]
        scale   = 0.00392
        
        print(width, height)
                
        # read pre-trained model
        net = cv.dnn.readNet(WEIGHTS, CONFIG)
        
        # create input blob  
        blob = cv.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        
        # set input blob for the network
        net.setInput(blob)
            
        # run inference through the network
        outs = net.forward(get_output_layers(net))
        
        # initialization
        class_ids       = []
        confidences     = []
        boxes           = []
        conf_threshold  = 0.1
        nms_threshold   = 0.4
        
        # find result
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(classes[class_id])
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
            
        # apply non-max suppression
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                
        # draw result
        for j in indices:
            box = boxes[j]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            if class_ids[j] not in skip_classes:
                cropper(IMAGE_PATH, x, y, x+w, y+h, i, j)
                detected +=1
                print('detected', detected)
    return detected
    
detected = yolo()
outfile = open('./output/temp_count/cropped_count.txt', "w")
outfile.writelines(str(detected))
outfile.close()
