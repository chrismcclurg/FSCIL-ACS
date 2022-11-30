# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================
    
from models.cbcl.functions import update_centroids, aff_simple
from models.cbcl.functions import save_var, load_var
import numpy as np
import random, time
import json

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
# UPDATE CLUSTER SPACE
f = open("./output/temp_count/current_train_item.txt", "r")
iClass = int(f.read())
nObsNewClass = [0 for x in range(nClassTotal)]

if (iClass > -1) and (iClass < nClassTotal):
    nObsNewClass[iClass] = 10

#classes for reporting when supply is out
CLASSES         = './utils/grocery.txt'
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#count new examples
nTrainNewClass = nObsNewClass.copy()
nObsTotClass = np.array(nObsTotClass)
nObsTotClass += np.array(nTrainNewClass)
nObsTotClass = list(nObsTotClass)

for i in range(len(nObsNewClass)):
    nObsTot += nObsNewClass[i]

if pLearner != 'SVM': #CBCL
    train_t0 = np.round(time.time(), 2)
    xTrainCurr = []
    yTrainCurr = []
    nObsNew = 0
    for iClass in range(len(nTrainNewClass)):
        for j in range(nTrainNewClass[iClass]):
            if len(yTrainTot[iClass]) > 0:
                xTrainObs = xTrainTot[iClass][0]
                yTrainObs = yTrainTot[iClass][0]
                del xTrainTot[iClass][0]
                del yTrainTot[iClass][0]
                xTrainCurr.append(xTrainObs)
                yTrainCurr.append(yTrainObs)
                nShotClass[yTrainObs] +=1
                nObsNew +=1
            else:
                # txt = 'Please remove {} from environment. Press Enter to continue.'.format(iClass)
                # temper = input(txt)                
                continue
            
    #count and write remaining
    filepath = './utils/grocery.txt'
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    classNo = [i for i in range(nClassTotal)]
    remCount = [len(yTrainTot[i]) for i in range(len(yTrainTot))]
    zipper = list(zip(classNo, remCount))
    np.random.shuffle(zipper)
    zipper.sort(key = lambda x: x[1])
    
    min_items_remaining = 'Minimum examples remaining: '
    for zippy in zipper[:5]:
        min_items_remaining += ' {} ({}), '.format(classes[zippy[0]], zippy[1])
               
    filepath = './output/temp_count/min_items_remaining.txt'
    f = open('./output/temp_count/min_items_remaining.txt', "w")
    f.writelines(min_items_remaining)
    f.close()
               
    #update centroids
    pack = [xTrainCurr, yTrainCurr, centClass, centWtClass, pDistLim, pDistMetric, covaClass, centStdClass]   
    [centClass, centWtClass, covaClass, centStdClass] = update_centroids(pack)  
        
    #count total centroids
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
    nObsNew = 0

    train_t0 = np.round(time.time(), 3)
    for iClass in range(len(nTrainNewClass)):
        for j in range(nTrainNewClass[iClass]):
            if len(yTrainTot[iClass]) > 0:
                xTrainObs = xTrainTot[iClass][0]
                yTrainObs = yTrainTot[iClass][0]
                xTrainBatch.append(xTrainObs)
                yTrainBatch.append(yTrainObs)
                del xTrainTot[iClass][0]
                del yTrainTot[iClass][0]
                nShotClass[yTrainObs] +=1
                nObsNew +=1
            else:
                # txt = 'Please remove {} from environment. Press Enter to continue.'.format(iClass)
                # temper = input(txt)
                continue   

    #count and write remaining
    filepath = './utils/grocery.txt'
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    min_items_remaining = 100.
    min_class_remaining = None
    for iClass in range(nClassTotal):
        nExampRem = len(yTrainTot[iClass])
        if nExampRem > 0 and nExampRem < min_items_remaining:
               min_items_remaining = nExampRem
               min_class_remaining = classes[iClass]
               
    temp = 'Minimum examples remaining:  {}, {}'.format(min_items_remaining, min_class_remaining)
    filepath = './output/temp_count/min_items_remaining.txt'
    f = open('./output/temp_count/min_items_remaining.txt', "w")
    f.writelines(temp)
    f.close()
        
#update affinity
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
        

            
            