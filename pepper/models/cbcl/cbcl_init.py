# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================
    
from sklearn.model_selection import train_test_split
from models.cbcl.functions import update_centroids, aff_simple
from models.cbcl.functions import save_var, incrementalData
import numpy as np
import random, pickle
import shutil, os
import json

VAR_PATH = './output/temp_cbcl/'
if os.path.exists(VAR_PATH): shutil.rmtree(VAR_PATH)
os.mkdir(VAR_PATH)

# -----------------------------------------------------------------------------
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
np.random.seed(pSeed)
random.seed(pSeed)

# -----------------------------------------------------------------------------
# INITIALIZE VARIABLES AND CLUSTER SPACE

#read visual features
readFile = './utils/features/'+ pDataName + '_' + pNetType + '_' + pNetFit + '_'
with open(readFile + 'train_features.data', 'rb') as fh: trainFeatRGB   = list(pickle.load(fh))
with open(readFile + 'test_features.data', 'rb') as fh:  testFeatRGB    = list(pickle.load(fh))
with open(readFile + 'train_labels.data', 'rb') as fh:   trainLabelRGB  = list(pickle.load(fh))
with open(readFile + 'test_labels.data', 'rb') as fh:    testLabelRGB   = list(pickle.load(fh))
pShotStart  = 41
pDistLim = 13

#filter by classes available for testing
import pandas as pd
data = pd.read_excel('./utils/grocery.xlsx')
fullID = list(range(81))
keepID = list(data['Old ID'])
fineID = list(data['New ID'])
coarseID = list(data['Coarse ID'])

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

def get_newLabels(oldLabels):
   prev = -1
   count = -1
   ans = []
   for i in range(len(oldLabels)):
       label = oldLabels[i]
       if label != prev:
           count +=1
           prev = label
       ans.append(count)  
   return np.array(ans)

def get_random_placement(coarseID, fineID):
    np.random.seed(0)
    random.seed(0)
    ans = []
    courseSet = list(set(coarseID))
    np.random.shuffle(courseSet)
    for x in courseSet:
        ix = find_indices(coarseID, x) 
        temp = []
        for ixx in ix:
           temp.append(fineID[ixx])
        np.random.shuffle(temp)
        ans.extend(temp)
    ans = [str(y) + '\n' for y in ans]
    return ans
               
count = 0
for i in range(len(fullID)):
    tempID = fullID[i]
    if tempID not in keepID:
        #remove from training
        ix_remove = find_indices(trainLabelRGB, tempID)
        for j in sorted(ix_remove, reverse=True): 
            del trainLabelRGB[j]
            del trainFeatRGB[j]
        #remove from testing
        ix_remove = find_indices(testLabelRGB, tempID)
        for j in sorted(ix_remove, reverse=True): 
            del testLabelRGB[j]
            del testFeatRGB[j]            

#re-order so that no numbers are skipped
trainLabelRGB   = get_newLabels(trainLabelRGB)
testLabelRGB    = get_newLabels(testLabelRGB)
trainFeatRGB    = np.array(trainFeatRGB)
testFeatRGB     = np.array(testFeatRGB)

#get placement of items
order = get_random_placement(coarseID, fineID)
outfile = open('./utils/grocery_placement.txt', "w")
outfile.writelines(order)
outfile.close()

#model discretized data
incData = incrementalData(trainFeatRGB, trainLabelRGB, testFeatRGB, testLabelRGB, pSeed)
incData.incFormat(1)
incTrainFeatures    = incData.incTrainFeatures
incTrainLabels      = incData.incTrainLabels
incTestFeatures     = incData.incTestFeatures
incTestLabels       = incData.incTestLabels
nClassTotal         = len(set(trainLabelRGB)) 

#shuffle by class
xTrainTot = [[] for i in range(nClassTotal)]
yTrainTot = [[] for i in range(nClassTotal)]
xTestTot  = [[] for i in range(nClassTotal)]
yTestTot  = [[] for i in range(nClassTotal)]
for iInc in range(len(incTrainLabels)):
    xTrainKeep, xTrainDel, yTrainKeep, yTrainDel = train_test_split(incTrainFeatures[iInc],incTrainLabels[iInc], random_state = pSeed, test_size=1)
    xTestKeep, xTestDel, yTestKeep, yTestDel = train_test_split(incTestFeatures[iInc],incTestLabels[iInc],random_state = pSeed, test_size=1)
    iClass = yTrainKeep[0]
    xTrainTot[iClass].extend(xTrainKeep)
    yTrainTot[iClass].extend(yTrainKeep)
    xTestTot[iClass].extend(xTestKeep)
    yTestTot[iClass].extend(yTestKeep) 
del xTrainKeep, yTrainKeep, xTestDel, yTestDel

#pull together all training and test instances for a dataset
for i in range(len(yTrainTot)):
    xTrainTot[i].extend(xTestTot[i])
    yTrainTot[i].extend(yTestTot[i])
del xTestTot, yTestTot
    
#prepare dataset for 90/10 train/test split
xTestTot = []
yTestTot = []
for iClass in range(nClassTotal):
    xTrainTot[iClass], xTestTemp, yTrainTot[iClass], yTestTemp = train_test_split(xTrainTot[iClass], yTrainTot[iClass], random_state = pSeed, test_size=0.10)
    xTestTot.extend(xTestTemp)
    yTestTot.extend(yTestTemp)
del xTestTemp, yTestTemp

#### move random open spot


#initialize
centClass       = [[] for x in range(nClassTotal)]      #centroids per class
centWtClass     = [[] for x in range(nClassTotal)]      #centroid wt per class
centStdClass    = [[] for x in range(nClassTotal)]      #centroid std per class
covaClass       = [[] for x in range(nClassTotal)]      #covariance matrices per class
nShotClass      = [0 for x in range(nClassTotal)]       #image count per class
weightClass     = [0 for x in range(nClassTotal)]       #weight per class
rAccClass       = [0 for x in range(nClassTotal)]       #test accuracy per class
xTrainBatch     = []                                    #features for SVM case                            
yTrainBatch     = []                                    #labels for SVM case
nObsNewClass    = [0 for x in range(nClassTotal)]       #observations per current increment

#initial examples
biasTemp        = [np.random.randint(1,10) for x in range(nClassTotal)]
biasClass       = [int(np.rint(x/ np.sum(biasTemp)*pShotStart)) for x in biasTemp]
needs_corrected = True
while needs_corrected:
    if np.sum(biasClass) > pShotStart:
        ix = random.randint(0,nClassTotal-1)
        if biasClass[ix] > 1 : biasClass[ix] -=1  
    elif np.sum(biasClass) < pShotStart:
        ix = random.randint(0,nClassTotal-1)
        biasClass[ix] +=1
    else:
        needs_corrected = False

#add at least one example per class
biasClass = [x+1 for x in biasClass]

if pLearner != 'SVM':  #CBCL
    xTrainCurr = []
    yTrainCurr = []   
    for iClass in range(len(biasClass)):
        for j in range(biasClass[iClass]):
            if len(xTrainTot[iClass])>0:
                xTrainCurr.append(xTrainTot[iClass][0])
                yTrainCurr.extend([yTrainTot[iClass][0]])
                nShotClass[iClass] +=1
                del xTrainTot[iClass][0]
                del yTrainTot[iClass][0]
            else:
                fReplace = True     #replace biased class with random
                while fReplace:
                    randClass = random.randint(0,nClassTotal-1)
                    if len(xTrainTot[randClass])>0:
                        xTrainCurr.append(xTrainTot[randClass][0])
                        yTrainCurr.extend([yTrainTot[randClass][0]])
                        nShotClass[randClass] +=1
                        del xTrainTot[randClass][0]
                        del yTrainTot[randClass][0]        
                        fReplace = False
                        
    #create centroids
    pack = [xTrainCurr, yTrainCurr, centClass, centWtClass, pDistLim, pDistMetric, covaClass, centStdClass]   
    [centClass, centWtClass, covaClass, centStdClass] = update_centroids(pack)   
        
    #count centroids
    nCentTotal = 0
    for iClass in range(nClassTotal): nCentTotal+=len(centClass[iClass]) 
        
    #find weights for fighting bias
    for iClass in range(nClassTotal):
        if nShotClass[iClass] != 0:
            weightClass[iClass] = np.divide(1,nShotClass[iClass])
        else:
            weightClass[iClass] = 0
    weightClass = np.divide(weightClass,np.sum(weightClass)) 

else: #SVM
    xNew = []
    yNew = []
    nObsNew = 0
    for iClass in range(len(biasClass)):
        for j in range(biasClass[iClass]):
            if len(yTrainTot[iClass]) > 0:
                xTrainObs = xTrainTot[iClass][0]
                yTrainObs = yTrainTot[iClass][0]
                xNew.append(xTrainTot[iClass][0])
                yNew.append(yTrainTot[iClass][0])
                xTrainBatch.append(xTrainObs)
                yTrainBatch.append(yTrainObs)
                del xTrainTot[iClass][0]
                del yTrainTot[iClass][0]
                nShotClass[yTrainObs] +=1
                nObsNew +=1
            else:
                fReplace = True     #replace biased class with random
                while fReplace:
                    randClass = random.randint(0,nClassTotal-1)
                    if len(xTrainTot[randClass])>0:
                        xTrainObs = xTrainTot[randClass][0]
                        yTrainObs = yTrainTot[randClass][0]
                        xNew.append(xTrainTot[randClass][0])
                        yNew.append(yTrainTot[randClass][0])
                        xTrainBatch.append(xTrainObs)
                        yTrainBatch.append(yTrainObs)
                        del xTrainTot[randClass][0]
                        del yTrainTot[randClass][0]
                        nShotClass[yTrainObs] +=1
                        nObsNew +=1        
                        fReplace = False
                        
    yTrainCurr = yTrainBatch.copy()

rAccClass0 = rAccClass.copy()

#initial counts
nObsTot         = 0
nObsTotClass    = [0 for i in range(nClassTotal)]
for i in range(len(yTrainCurr)): 
    nObsTot +=1
    iClass = yTrainCurr[i]
    nObsTotClass[iClass] +=1
    
#initialize output variables
final_obs       = []
final_acc       = []
final_runTime   = []
final_runDist   = []
final_trainTime = []
final_predRate  = []
final_status    = 'IP'
runTime         = 0
runDist         = 0
trainTime       = 0
iInc            = 0

#initial affinity
aClass = aff_simple(pBias, centWtClass, centStdClass, rAccClass, rAccClass0, pMod)

# -----------------------------------------------------------------------------
# save data
save_var(iInc, 'iInc')

save_var(xTrainTot, 'xTrainTot')
save_var(yTrainTot, 'yTrainTot')
save_var(xTestTot, 'xTestTot')
save_var(yTestTot, 'yTestTot')
save_var(xTrainBatch, 'xTrainBatch') 
save_var(yTrainBatch, 'yTrainBatch')

save_var(centClass, 'centClass')
save_var(centWtClass, 'centWtClass')
save_var(centStdClass, 'centStdClass')
save_var(covaClass, 'covaClass')
save_var(weightClass, 'weightClass')
save_var(nShotClass, 'nShotClass')
save_var(aClass, 'aClass')

save_var(rAccClass0, 'rAccClass0')
save_var(rAccClass, 'rAccClass')
save_var(nObsTotClass, 'nObsTotClass')
save_var(nObsTot, 'nObsTot')
save_var(runTime, 'runTime')
save_var(runDist, 'runDist')
save_var(trainTime, 'trainTime')

save_var(final_status, 'final_status')
save_var(final_acc, 'final_acc')
save_var(final_runTime, 'final_runTime')
save_var(final_runDist, 'final_runDist')
save_var(final_trainTime, 'final_trainTime')
save_var(final_obs, 'final_obs')
save_var(final_predRate, 'final_predRate')

#create blank temp files
with open('./output/temp_count/guess_true_record.txt', 'w') as f:   pass
with open('./output/temp_count/current_train_item.txt', 'w') as f:   pass
with open('./output/temp_count/cumulative_distance.txt', 'w') as f:   pass
with open('./output/temp_count/cumulative_runTime.txt', 'w') as f:   pass
with open('./output/temp_count/cumulative_trainTime.txt', 'w') as f:   pass
with open('./output/temp_count/min_items_remaining.txt', 'w') as f:   pass
with open('./output/temp_count/cropped_count.txt', 'w') as f:   pass
with open('./output/temp_count/current_acc.txt', 'w') as f:   pass
with open('./output/temp_count/images/2_guesses/labels.txt', 'w') as f:   pass







