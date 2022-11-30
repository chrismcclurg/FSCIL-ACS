# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================

# imports
import numpy as np
from classification_models.tfkeras import Classifiers
from keras.models import Model
from keras.utils import image_utils as image
from cv2 import resize
from PIL import Image
import shutil
import os
import random
import json
from models.cbcl.functions import load_var, CBCL_WVS, CBCL_SVM, CBCL_PR, SVM_simple

# set no GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
     
# model parameters
with open('./utils/param.txt') as f: params = f.read()
params          = json.loads(params)
pFileNo         = params['pFileNo']
pMod            = params['pMod']   
pSeed           = params['pSeed']
pDataName       = params['pDataName']
pLearner        = params['pLearner']                  
pBias           = params['pBias']
pNetType        = params['pNetType']           
pNetFit         = params['pNetFit']          
pDistMetric     = params['pDistMetric']         
pCentroidPred   = params['pCentroidPred'] 
nClassTotal     = params['nClassTotal']
pDistLim        = params['pDistLim']
pInc            = params['pInc']
# np.random.seed(pSeed)
# random.seed(pSeed)
# -----------------------------------------------------------------------------
#load data
iInc            = load_var('iInc')

xTrainTot       = load_var('xTrainTot')
yTrainTot       = load_var('yTrainTot')
xTestTot        = load_var('xTestTot')
yTestTot        = load_var('yTestTot')
xTrainBatch     = load_var('xTrainBatch') 
yTrainBatch     = load_var('yTrainBatch')

centClass       = load_var('centClass')
centWtClass     = load_var('centWtClass')
centStdClass    = load_var('centStdClass')
covaClass       = load_var('covaClass')
weightClass     = load_var('weightClass')
nShotClass      = load_var('nShotClass')
aClass          = load_var('aClass')

rAccClass0      = load_var('rAccClass0')
rAccClass       = load_var('rAccClass')
nObsTotClass    = load_var('nObsTotClass')
nObsTot         = load_var('nObsTot')
runTime         = load_var('runTime')
trainTime       = load_var('trainTime')
runDist         = load_var('runDist')

final_status    = load_var('final_status')
final_acc       = load_var('final_acc')
final_runTime   = load_var('final_runTime')
final_runDist   = load_var('final_runDist')
final_trainTime = load_var('final_trainTime')
final_obs       = load_var('final_obs')
final_predRate  = load_var('final_predRate')


# -----------------------------------------------------------------------------
# MAKE PREDICTIONS (but no accuracy recovered)
CLASSES         = './utils/grocery.txt'
INPUT_PATH      = './output/temp_count/images/1_cropped/'
OUTPUT_PATH     = './output/temp_count/images/2_guesses/'
net, PINPUT     = Classifiers.get('resnet34')
k_model         = net(input_shape=(224,224,3), weights='imagenet')
MODEL           = Model(inputs = k_model.input, outputs = k_model.get_layer('pool1').output)

with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
def read_images():
    imagesInput = os.listdir(INPUT_PATH)
    images = [] 
    for i in range(len(imagesInput)):
        IMAGE_PATH = INPUT_PATH + imagesInput[i]
        im = Image.open(IMAGE_PATH)
        im = np.array(im)
        im = resize(im, (348,348))
        images.append(im)
    return images

def extract_features(images):
    ans  = []    
    for i in range(0,len(images)):
        img = resize(images[i],(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = PINPUT(x)
        features = MODEL.predict(x)
        features_np = np.array(features)
        features_f = features_np.flatten()
        ans.append(features_f)
    return ans

#model setup
net, ppInput   = Classifiers.get('resnet34')
k_model        = net(input_shape=(224,224,3), weights='imagenet')
MODEL          = Model(inputs = k_model.input, outputs = k_model.get_layer('pool1').output)

#data from cropped
images = read_images()

if len(images) > 0:

    #extract features
    features = extract_features(images) 
    
    #make new predictions
    xTestTot        = features.copy()
    yTestTot        = [0 for ix in range(len(features))]
    if pLearner != 'SVM':  #CBCL
        pack = [xTestTot, yTestTot, centClass, pCentroidPred, nClassTotal, weightClass, pDistMetric, covaClass, centWtClass]
        if pLearner == 'CBCLWVS':       _, _, predicted_labels = CBCL_WVS(pack)
        elif pLearner == 'CBCLSVM':     _, _, predicted_labels = CBCL_SVM(pack)
        else:                           _, _, predicted_labels = CBCL_PR(pack)
    else: #SVM 
        _, _, predicted_labels  = SVM_simple(xTrainBatch, yTrainBatch, xTestTot, yTestTot, nClassTotal, 'SVM', pSeed)   
    
#write predictions to file
if os.path.exists(OUTPUT_PATH): shutil.rmtree(OUTPUT_PATH)
os.mkdir(OUTPUT_PATH)
ans = [] 
for i in range(0, len(images)):    
    classNo = predicted_labels[i]
    temp = '({})\t {}\t {}\n'.format(classNo, classes[classNo], aClass[classNo])
    ans.extend(temp)
outfile = open(OUTPUT_PATH + 'labels.txt', "w")
outfile.writelines(ans)
outfile.close()