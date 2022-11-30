# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================

import pickle
import random
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from copy import deepcopy

def find_distance(data_vec,centroid,distance_metric):
    if distance_metric=='euclidean':
        return np.linalg.norm(data_vec-centroid)
    elif distance_metric == 'euclidean_squared':
        return np.square(np.lnalg.norm(data_vec-centroid))
    elif distance_metric == 'cosine':
        return distance.cosine(data_vec,centroid)

def predict_multiple(data_vec,centroids,distance_metric,pCentroidPred,weighting):
    dist = []
    for iClass in range(len(centroids)):
        temp = predict_multiple_class(data_vec,centroids[iClass], iClass, distance_metric)
        dist.extend(temp)
    sorted_dist = sorted(dist)
    common_classes = [0]*len(centroids)
    if pCentroidPred > len(sorted_dist): pCentroidPred = len(sorted_dist)
    for i in range(0,pCentroidPred):
        if sorted_dist[i][0]==0.0:
            common_classes[sorted_dist[i][1]] += 1
        else:
            common_classes[sorted_dist[i][1]] += ((1/(i+1))*
                                                ((sorted_dist[len(sorted_dist)-1][0]-sorted_dist[i][0])/(sorted_dist[len(sorted_dist)-1][0]-sorted_dist[0][0])))
    common_classes = np.multiply(common_classes, weighting)
    return np.argmax(common_classes)

def predict_multiple_class(data_vec,centroids,iClass,distance_metric):
    dist = [[0,iClass] for x in range(len(centroids))]
    for i in range(0,len(centroids)):
        dist[i][0] = find_distance(data_vec,centroids[i],distance_metric)
    return dist

def CBCL_WVS(pack):
    xTestTot            = pack[0]
    yTestTot            = pack[1]
    centClass           = pack[2]
    pCentroidPred       = pack[3]
    nClassTotal         = pack[4]
    weightClass         = pack[5]
    distance_metric     = pack[6]

    centroids   = [x for x in centClass if x]
    weighting   = [x for x in weightClass if x]
    yPred       = -1
    nLabel      = [0 for x in range(nClassTotal)] 
    rAcc        = 0
    rAccClass   = [0 for x in range(nClassTotal)] 
    classTest   = list(np.unique(yTestTot))
    yPreds      = []
    
    for i in range(0,len(yTestTot)):
        iClass  = classTest.index(yTestTot[i])                
        yPred   = predict_multiple(xTestTot[i],centroids,distance_metric, pCentroidPred, weighting)
        nLabel[iClass]+=1
        if yPred == iClass: rAccClass[iClass]+=1
        yPreds.append(yPred)
            
    for i in range(nClassTotal):
        if nLabel[i]>0:     
            rAccClass[i] = rAccClass[i]/nLabel[i]
        
    rAcc = accuracy_score(yPreds, yTestTot) #true test accuracy
    rAcc = np.round(rAcc,3) 
    return rAcc, rAccClass, yPred

def CBCL_SVM(pack):
    
    xTestTot    = pack[0]
    yTestTot    = pack[1]
    centClass   = pack[2]
    nClassTotal = pack[4]
    
    xTrain      = []
    yTrain      = []

    for iClass in range(len(centClass)):
        for jVec in range(len(centClass[iClass])):
            vec = centClass[iClass][jVec]
            lab = iClass
            xTrain.append(vec)
            yTrain.append(lab)
    
    model = SVC(kernel = 'linear')
    nCorrClass  = [0 for x in range(nClassTotal)]
    nActClass   = [0 for x in range(nClassTotal)]
    rAccClass   = [0 for x in range(nClassTotal)]
   
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTestTot)
    rAcc = accuracy_score(yPred, yTestTot)
    rAcc = np.round(rAcc,3) 
    
    for i in range(len(yTestTot)):
        iClass = yTestTot[i] 
        predVal = yPred[i]
        nActClass[iClass] +=1
        if predVal == iClass: nCorrClass[iClass] +=1
    
    for iClass in range(nClassTotal):
            if nActClass[iClass] != 0:   
                rAccClass[iClass] = np.round(np.divide(nCorrClass[iClass], nActClass[iClass]), 3)
    
    return rAcc, rAccClass, yPred

def CBCL_PR(pack):
    
    xTestTot    = pack[0]
    yTestTot    = pack[1]
    centClass   = pack[2]
    nClassTotal = pack[4]
    covaClass   = pack[7]
    centWtClass = pack[8]
    
    # generate pseduo exemplars from the centroids
    xTrain = []
    yTrain = []
    nPsuedoPerClass = 40
    
    for iClass in range(nClassTotal):
        ones_count = centWtClass[iClass].count(1)
        req_samples = nPsuedoPerClass - ones_count
        if len(centWtClass[iClass])-ones_count>0:
            how_many_per_centroid = round(req_samples/(len(centWtClass[iClass])-ones_count))
        for jCent in range(len(centWtClass[iClass])):
            if centWtClass[iClass][jCent]>1:
                print(centWtClass[iClass][jCent])
                temp = list(np.random.multivariate_normal(centClass[iClass][jCent],covaClass[iClass][jCent],how_many_per_centroid))
                xTrain.extend(temp)
                yTrain.extend([iClass for x in range(how_many_per_centroid)])
            else:
                xTrain.append(centClass[iClass][jCent])
                yTrain.append(iClass)
    

    model = SVC(kernel = 'linear')
    nCorrClass  = [0 for x in range(nClassTotal)]
    nActClass   = [0 for x in range(nClassTotal)]
    rAccClass   = [0 for x in range(nClassTotal)]
   
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTestTot)
    rAcc = accuracy_score(yPred, yTestTot)
    rAcc = np.round(rAcc,3) 
    
    for i in range(len(yTestTot)):
        iClass = yTestTot[i] 
        predVal = yPred[i]
        nActClass[iClass] +=1
        if predVal == iClass: nCorrClass[iClass] +=1
    
    for iClass in range(nClassTotal):
            if nActClass[iClass] != 0:   
                rAccClass[iClass] = np.round(np.divide(nCorrClass[iClass], nActClass[iClass]), 3)
    
    return rAcc, rAccClass, yPred


def update_centroids(pack):
    xTrainCurr          = pack[0]
    yTrainCurr          = pack[1]
    centClass           = pack[2]
    centWtClass         = pack[3]
    pDistLim            = pack[4]
    pDistMetric         = pack[5]
    covaClass           = pack[6]
    centStdClass        = pack[7]

    for i in range(len(yTrainCurr)):
        iClass          = yTrainCurr[i]
        tempXTrain      = xTrainCurr[i].copy()
        tempCentroids   = centClass[iClass].copy()
        tempCentWts     = centWtClass[iClass].copy()
        tempCovas       = covaClass[iClass].copy()
        tempStds        = centStdClass[iClass].copy()
        tempDist        = []
        tempIxCent      = []
        
        for j in range(0,len(tempCentroids)):
            d = find_distance(tempXTrain,tempCentroids[j],pDistMetric)
            if d < pDistLim:
                tempDist.append(d)
                tempIxCent.append(j)
        
        if len(tempDist)==0:        
            #create new cluster
            tempCentroids.append(tempXTrain)
            tempCentWts.append(1)
            tempCovas.append(np.identity(len(tempXTrain)))
            tempStds.append(np.zeros(512,))
            # tempStds.append(0.0)

        else:                       
            #identify closest cluster
            ixMinDist = np.argmin(tempDist)
            jx = tempIxCent[ixMinDist]
            
            #online covariance update
            for_cov = list(np.random.multivariate_normal(tempCentroids[jx],tempCovas[jx],tempCentWts[jx]))
            for_cov.append(tempXTrain)
            tempCovas[jx] = np.cov(np.array(for_cov).T)
            
            #online mean update
            m0 = tempCentroids[jx].copy()
            tempCentroids[jx] = np.add(np.multiply(tempCentWts[jx],tempCentroids[jx]),tempXTrain)
            tempCentWts[jx]+=1
            tempCentroids[jx] = np.divide(tempCentroids[jx],(tempCentWts[jx]))
            
            #online std update (this is really variance)
            s0 = tempStds[jx]
            xk = tempXTrain.copy()
            k = tempCentWts[jx] 
            mk = m0 + (xk - m0)/k            
            sk = s0 + (xk - m0)*(xk - mk)
            vk = sk / k
            tempStds[jx] = vk
        
        centClass[iClass]   = tempCentroids
        centWtClass[iClass] = tempCentWts
        covaClass[iClass]    = tempCovas
        centStdClass[iClass] = tempStds
        
    return [centClass, centWtClass, covaClass, centStdClass]


def aff_simple(pBiasType, centWtClass, centStdClass, rAccClass, rAccClass0, pMod):
    
    nClassTotal = len(centWtClass)
    working     = [0 for x in range(nClassTotal)]
    pTopBot     = int(np.round(nClassTotal/4))
    
    #random
    if pBiasType == 'uniform':
        working = [random.randint(0,100) for x in range(nClassTotal)]
    
    #lowest accuracy
    elif pBiasType == 'inverse':
        accFix = np.min([x for x in rAccClass if x>0])
        for iClass in range(nClassTotal):
            if rAccClass[iClass] == 0: temp = 1/accFix
            else: temp = 1/rAccClass[iClass]
            working[iClass] = temp
    
    #highest change of accuracy
    elif pBiasType == 'accuracy':
        if len(rAccClass0) == len(rAccClass):
            for iClass in range(nClassTotal):
                working[iClass] = (rAccClass[iClass] - rAccClass0[iClass])
        else: 
            working = [random.randint(0,100) for x in range(nClassTotal)]

    #lowest cluster weight
    elif pBiasType == 'clusterWt':
        clusterWt = [np.mean(x) if len(x) > 0 else 0 for x in centWtClass]
        clusterFix = np.min([x for x in clusterWt if x>0])
        working = [1/x if x!=0 else 1/clusterFix for x in clusterWt]

    #lowest class weight
    elif pBiasType == 'classWt':
        classWt = [np.sum(x) if len(x) > 0 else 0 for x in centWtClass]
        classFix = np.min([x for x in classWt if x>0])
        working = [1/x if x!=0 else 1/classFix for x in classWt]
        
    #lowest std
    elif pBiasType == 'clusterStdLow':
        clusterStd = [np.mean(np.mean(x)) if len(x) > 0 else 0 for x in centStdClass]
        classFix = np.min([x for x in clusterStd if x>0])
        working = [1/x if x!=0 else 1/classFix for x in clusterStd]
    
    #highest std
    elif pBiasType == 'clusterStdHigh':
        clusterStd = [np.mean(np.mean(x)) if len(x) > 0 else 0 for x in centStdClass]
        classFix = np.min([x for x in clusterStd if x>0])
        working = [x if x!=0 else classFix for x in clusterStd]           

    #aff (random but ordered by working)
    aClass = [0 for i in range(nClassTotal)]
    classNo = [i for i in range(nClassTotal)]
    zipper = list(zip(classNo, working))
    np.random.shuffle(zipper)
    zipper.sort(key = lambda x: x[1], reverse = True)
    val = -1*nClassTotal      
    for zippy in zipper:
        ix = zippy[0]
        aClass[ix] = val 
        val +=1
    
    # #least curious (25%)
    # tempIx = list(np.argsort(np.array(working))[0:pTopBot])
    # if pMod == 0: 
    #     for ix in tempIx: aClass[ix] = int(20)
    # elif pMod == 1: 
    #     for ix in tempIx: aClass[ix] = int(0)
    # elif pMod == 2: 
    #     for ix in tempIx: aClass[ix] = int(5)
    # elif pMod == 3: 
    #     for ix in tempIx: aClass[ix] = int(-5)    
    # elif pMod == 4: 
    #     for ix in tempIx: aClass[ix] = int(-10)
    # elif pMod == 5: 
    #     for ix in tempIx: aClass[ix] = int(-20)  
        
    # #2nd-least curious (25%)
    # tempIx = list(np.argsort(np.array(working))[pTopBot: (2*pTopBot)])
    # if pMod == 0: 
    #     for ix in tempIx: aClass[ix] = int(10)
    # elif pMod == 1: 
    #     for ix in tempIx: aClass[ix] = int(0)
    # elif pMod == 2: 
    #     for ix in tempIx: aClass[ix] = int(5)
    # elif pMod == 3: 
    #     for ix in tempIx: aClass[ix] = int(-5)
    # elif pMod == 4: 
    #     for ix in tempIx: aClass[ix] = int(-10)
    # elif pMod == 5: 
    #     for ix in tempIx: aClass[ix] = int(-20)
    
    # #2nd-most curious (25%)
    # tempIx = list(np.argsort(np.array(working))[(nClassTotal- 2*pTopBot): (nClassTotal- pTopBot)])
    # if pMod == 5: 
    #     for ix in tempIx: aClass[ix] = int(-20)
    # else:
    #     for ix in tempIx: aClass[ix] = int(-10)
    
    # #most curious (25%)
    # tempIx = list(np.argsort(np.array(working))[(nClassTotal- pTopBot): nClassTotal])
    # for ix in tempIx: aClass[ix] = int(-20)
    
    #instead of quartiles just give weights based on working order

    
    return aClass

def aff_redistrict(nShotClass, redClass, iInc, pMod):
    
    nClassTotal = len(redClass)
    working     = [0 for x in range(nClassTotal)]
    pTopBot     = int(np.round(nClassTotal/4))
    
    lenRed = len([x for x in redClass if x>0])
          
    if iInc >1 and lenRed >0: #redistrict
        nc = [x/ np.sum(nShotClass) for x in nShotClass]
        ncFix = np.min([x for x in nc if x>0])
        nc = [x if x > 0 else ncFix for x in nc]
        for iClass in range(nClassTotal):
           working[iClass] = redClass[iClass] / nc[iClass]
           
    else: #random (essentially uniform in the batch case)
        working = [random.randint(0,100) for x in range(nClassTotal)]
        
    #aff
    aClass = [0 for i in range(nClassTotal)]
    
    #least curious (25%)
    tempIx = list(np.argsort(np.array(working))[0:pTopBot])
    if pMod == 0: 
        for ix in tempIx: aClass[ix] = int(20)
    elif pMod == 1: 
        for ix in tempIx: aClass[ix] = int(0)
    elif pMod == 2: 
        for ix in tempIx: aClass[ix] = int(5)
    elif pMod == 3: 
        for ix in tempIx: aClass[ix] = int(-5)    
    elif pMod == 4: 
        for ix in tempIx: aClass[ix] = int(-10)
    elif pMod == 5: 
        for ix in tempIx: aClass[ix] = int(-20)  
        
    #2nd-least curious (25%)
    tempIx = list(np.argsort(np.array(working))[pTopBot: (2*pTopBot)])
    if pMod == 0: 
        for ix in tempIx: aClass[ix] = int(10)
    elif pMod == 1: 
        for ix in tempIx: aClass[ix] = int(0)
    elif pMod == 2: 
        for ix in tempIx: aClass[ix] = int(5)
    elif pMod == 3: 
        for ix in tempIx: aClass[ix] = int(-5)
    elif pMod == 4: 
        for ix in tempIx: aClass[ix] = int(-10)
    elif pMod == 5: 
        for ix in tempIx: aClass[ix] = int(-20)
    
    #2nd-most curious (25%)
    tempIx = list(np.argsort(np.array(working))[(nClassTotal- 2*pTopBot): (nClassTotal- pTopBot)])
    if pMod == 5: 
        for ix in tempIx: aClass[ix] = int(-20)
    else:
        for ix in tempIx: aClass[ix] = int(-10)
    
    #most curious (25%)
    tempIx = list(np.argsort(np.array(working))[(nClassTotal- pTopBot): nClassTotal])
    for ix in tempIx: aClass[ix] = int(-20)

    return aClass
def SVM_simple(xTrainBatch, yTrainBatch, xTestTot, yTestTot, nClassTotal, pACSType, pSeed): 
    
    model = SVC(kernel = 'linear')
    nCorrClass  = [0 for x in range(nClassTotal)]
    nActClass   = [0 for x in range(nClassTotal)]
    rAccClass   = [0 for x in range(nClassTotal)]
   
    model.fit(xTrainBatch, yTrainBatch)
    yPred = model.predict(xTestTot)
    rAcc = accuracy_score(yPred, yTestTot)
    rAcc = np.round(rAcc,3) 
    
    for i in range(len(yTestTot)):
        iClass = yTestTot[i] 
        predVal = yPred[i]
        nActClass[iClass] +=1
        if predVal == iClass: nCorrClass[iClass] +=1
    
    for iClass in range(nClassTotal):
            if nActClass[iClass] != 0:   
                rAccClass[iClass] = np.round(np.divide(nCorrClass[iClass], nActClass[iClass]), 3)
    
    return rAcc, rAccClass, yPred

def SVM_redistrict(xTrainBatch, yTrainBatch, xNew, yNew, xTestTot, yTestTot, prevSplit, redClass, pACSType, pSeed, iInc): 
    
    nClassTotal = len(redClass)
    kf          = KFold(n_splits = 10, random_state = pSeed, shuffle = True)
    model       = SVC(kernel = 'linear')
    nCorrClass  = [0 for x in range(nClassTotal)]
    nActClass   = [0 for x in range(nClassTotal)]
    rAccClass   = [0 for x in range(nClassTotal)]
    redClass    = [0 for x in range(nClassTotal)]   #recount for every round
    
    if iInc == 0:
        prevSplit   = [] 
        for trainIx, testIx in kf.split(xTrainBatch):
            xTrainCV, xTestCV = np.array(xTrainBatch)[trainIx], np.array(xTrainBatch)[testIx]
            yTrainCV, yTestCV = np.array(yTrainBatch)[trainIx], np.array(yTrainBatch)[testIx]
            prevSplit.append(testIx)
    
            model.fit(xTrainCV, yTrainCV)
            predVal = model.predict(xTestCV)
            
            for k in range(len(yTestCV)): 
                pred = predVal[k]
                iClass = yTestCV[k]
                nActClass[iClass] +=1
                if pred == iClass: nCorrClass[iClass] +=1
                
        for iClass in range(nClassTotal):
            if nActClass[iClass] != 0:   
                rAccClass[iClass] = np.round(np.divide(nCorrClass[iClass], nActClass[iClass]), 3)
          
    else: 
        iF = 0
        if len(prevSplit) > 0: lenOld = np.max([np.max(x) for x in prevSplit]) + 1
        else: lenOld = 0
        
        for trainIx, testIx in kf.split(xNew):
            
            x1minFnew, xFnew = np.array(xNew)[trainIx], np.array(xNew)[testIx]
            y1minFnew, yFnew = np.array(yNew)[trainIx], np.array(yNew)[testIx]
                        
            mask = np.ones((len(xTrainBatch), ), dtype = bool)
            mask[prevSplit[iF]] = False
            x1minFold = np.array(xTrainBatch)[mask]
            y1minFold = np.array(yTrainBatch)[mask]
            xFold = np.array(xTrainBatch)[prevSplit[iF]]
            yFold = np.array(yTrainBatch)[prevSplit[iF]]
            
            xTrainCV = np.append(x1minFold, x1minFnew, axis = 0)
            yTrainCV = np.append(y1minFold, y1minFnew, axis = 0)
            xTestCV = np.append(xFold, xFnew, axis = 0)
            yTestCV = np.append(yFold, yFnew, axis = 0)
            
            model.fit(x1minFold, y1minFold)
            predOld = model.predict(xTestCV)
            
            model.fit(xTrainCV, yTrainCV)
            predNew = model.predict(xTestCV)
            
            tempIx = list(testIx + lenOld)
            prevSplit[iF] = np.append(prevSplit[iF], tempIx)
            iF+=1
            
            for i in range(len(yTestCV)):
                pred0 = predOld[i]
                pred1 = predNew[i]
                iClass = yTestCV[i]   
                if pred1 != pred0:  redClass[iClass] +=1
        
    model.fit(xTrainBatch, yTrainBatch)
    yPred = model.predict(xTestTot)
    rAcc = accuracy_score(yPred, yTestTot)
    rAcc = np.round(rAcc,3) 
    
    for i in range(len(yTestTot)):
        iClass = yTestTot[i] 
        predVal = yPred[i]
        nActClass[iClass] +=1
        if predVal == iClass: nCorrClass[iClass] +=1
    
    for iClass in range(nClassTotal):
            if nActClass[iClass] != 0:   
                rAccClass[iClass] = np.round(np.divide(nCorrClass[iClass], nActClass[iClass]), 3) 
    return rAcc, rAccClass, prevSplit, redClass, yPred

def save_var(x1, name):
    path = './output/temp_cbcl/'
    with open(path + name + '.data', 'wb') as fh:   pickle.dump(x1, fh)

def load_var(name):
    try:
        filepath = './output/temp_cbcl/' + name + '.data'
        with open(filepath, 'rb') as fh:   x1 = pickle.load(fh)
        return x1
    except EOFError:
        return []
    
class incrementalData:
    def __init__(self, trainFeatRGB, trainLabelRGB, testFeatRGB, testLabelRGB, pSeed):
        self.trainFeatRGB       = trainFeatRGB
        self.testFeatRGB        = testFeatRGB
        self.trainLabelRGB      = trainLabelRGB
        self.testLabelRGB       = testLabelRGB       
        self.nClassTotal        = len(set(trainLabelRGB))        
        self.seed               = pSeed
        self.incTrainFeatures   = None
        self.incTrainLabels     = None
        self.incTestFeatures    = None
        self.incTestLabels      = None
        np.random.seed(self.seed)
        random.seed(self.seed)

    def incFormat(self, nClassPerInc):
        
        nClassTotal = len(set(self.trainLabelRGB))
        
        #initialize shuffled data
        shuf_trainLabelRGB  = deepcopy(self.trainLabelRGB)
        shuf_testLabelRGB   = deepcopy(self.testLabelRGB)
 
        #order data by class
        ixSortTrain     = np.argsort(shuf_trainLabelRGB)        
        ixSortTest      = np.argsort(shuf_testLabelRGB)
        
        sort_trainFeatRGB   = self.trainFeatRGB[ixSortTrain[::1]]
        sort_testFeatRGB    = self.testFeatRGB[ixSortTest[::1]]      
        sort_trainLabelRGB  = shuf_trainLabelRGB[ixSortTrain[::1]]
        sort_testLabelRGB   = shuf_testLabelRGB[ixSortTest[::1]]
       
        #sort data by class     
        trainFeatByClass    = [[] for iClass in range(nClassTotal)]  
        testFeatByClass     = [[] for iClass in range(nClassTotal)]  
        trainLabelByClass   = [[] for iClass in range(nClassTotal)]  
        testLabelByClass    = [[] for iClass in range(nClassTotal)]  
        
        for iImg in range(len(sort_trainLabelRGB)):
            for iClass in range(nClassTotal): 
                if sort_trainLabelRGB[iImg] == iClass:
                    trainLabelByClass[iClass].append(sort_trainLabelRGB[iImg])
                    trainFeatByClass[iClass].append(sort_trainFeatRGB[iImg])
                    
        for iImg in range(len(sort_testLabelRGB)):
            for iClass in range(nClassTotal): 
                if sort_testLabelRGB[iImg] == iClass:
                    testLabelByClass[iClass].append(sort_testLabelRGB[iImg])
                    testFeatByClass[iClass].append(sort_testFeatRGB[iImg])
        
        #shuffle data by class
        shuffClass = list(range(nClassTotal))
        random.shuffle(shuffClass)
        shuffTrainFeatures  = []
        shuffTrainLabels    = []
        shuffTestFeatures   = []
        shuffTestLabels     = []  
                      
        
        for iClass in range(nClassTotal): 
            for iImg in range(len(sort_trainLabelRGB)):
                if sort_trainLabelRGB[iImg] == shuffClass[iClass]:
                    shuffTrainLabels.append(sort_trainLabelRGB[iImg])
                    shuffTrainFeatures.append(sort_trainFeatRGB[iImg])

        for iClass in range(nClassTotal): 
            for iImg in range(len(sort_testLabelRGB)):
                if sort_testLabelRGB[iImg] == shuffClass[iClass]:
                    shuffTestLabels.append(sort_testLabelRGB[iImg])
                    shuffTestFeatures.append(sort_testFeatRGB[iImg])
        
        trainFeatRGB    = shuffTrainFeatures
        testFeatRGB     = shuffTestFeatures
        trainLabelRGB   = shuffTrainLabels
        testLabelRGB    = shuffTestLabels

        #divide data into increments
        if nClassPerInc != None:
            if nClassPerInc > nClassTotal: nClassPerInc = self.nClassTotal
            nIncTotal           = int(self.nClassTotal/nClassPerInc)
            incTrainFeatures    = [[] for x in range(nIncTotal)]
            incTrainLabels      = [[] for x in range(nIncTotal)]
            incTestFeatures     = [[] for x in range(nIncTotal)]
            incTestLabels       = [[] for x in range(nIncTotal)]

            for iInc in range(nIncTotal):
                classCurrent = range(0+iInc*nClassPerInc, nClassPerInc+iInc*nClassPerInc)
                for iClass in range(len(classCurrent)): 
                    tempClass = shuffClass[classCurrent[iClass]]
                    for iImg in range(len(trainLabelRGB)):
                        if trainLabelRGB[iImg] == tempClass:
                            incTrainLabels[iInc].append(trainLabelRGB[iImg])
                            incTrainFeatures[iInc].append(trainFeatRGB[iImg])
                            
                    for iImg in range(len(testLabelRGB)):
                        if testLabelRGB[iImg] == tempClass:
                            incTestLabels[iInc].append(testLabelRGB[iImg])
                            incTestFeatures[iInc].append(testFeatRGB[iImg])
                            
        else:
            incTrainFeatures    = [[trainFeatRGB]]
            incTrainLabels      = [[trainLabelRGB]]
            incTestFeatures     = [[testFeatRGB]]
            incTestLabels       = [[testLabelRGB]]
            
        self.incTrainFeatures   = incTrainFeatures
        self.incTrainLabels     = incTrainLabels
        self.incTestFeatures    = incTestFeatures
        self.incTestLabels      = incTestLabels
               
 