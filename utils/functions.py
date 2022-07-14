# =============================================================================
# Incremental Learning (CBCL) with Active Class Selection
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer
# =============================================================================

import numpy as np
from scipy.spatial import distance
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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
    return rAcc, rAccClass

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
    
    return rAcc, rAccClass

def update_centroids(pack):
    xTrainCurr          = pack[0]
    yTrainCurr          = pack[1]
    centClass           = pack[2]
    centWtClass         = pack[3]
    pDistLim            = pack[4]
    pDistMetric         = pack[5]

    for i in range(len(yTrainCurr)):
        iClass          = yTrainCurr[i]
        tempXTrain      = xTrainCurr[i].copy()
        tempCentroids   = centClass[iClass].copy()
        tempCentWts     = centWtClass[iClass].copy()
        tempDist        = []
        tempIxCent      = []
        
        for j in range(0,len(tempCentroids)):
            d = find_distance(tempXTrain,tempCentroids[j],pDistMetric)
            if d < pDistLim:
                tempDist.append(d)
                tempIxCent.append(j)
        if len(tempDist)==0:
            tempCentroids.append(tempXTrain)
            tempCentWts.append(1)

        else:
            ixMinDist = np.argmin(tempDist)
            jx = tempIxCent[ixMinDist]
            tempCentroids[jx] = np.add(np.multiply(tempCentWts[jx],tempCentroids[jx]),tempXTrain)
            tempCentWts[jx]+=1
            tempCentroids[jx] = np.divide(tempCentroids[jx],(tempCentWts[jx]))
        
        centClass[iClass]   = tempCentroids
        centWtClass[iClass] = tempCentWts
        
    return [centClass, centWtClass]


def aff_simple(pBiasType, centWtClass, rAccClass, rAccClass0, pMod):
    
    nClassTotal = len(centWtClass)
    working     = [0 for x in range(nClassTotal)]
    pTopBot     = int(np.round(nClassTotal/4))
    
    #random
    if pBiasType == 'random':
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
    
    #uniform 
    elif pBiasType == 'uniform':
        working = [1 for x in range(nClassTotal)]
        pMod = 5
        

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
    
    return rAcc, rAccClass

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
    return rAcc, rAccClass, prevSplit, redClass