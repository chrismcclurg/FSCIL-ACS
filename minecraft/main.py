# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================
    
from multiprocess import Process, Queue
from sklearn.model_selection import train_test_split
from models.cbcl.functions import CBCL_WVS, CBCL_SVM, CBCL_PR, SVM_redistrict, SVM_simple
from models.cbcl.functions import update_centroids, aff_simple, aff_redistrict
from models.cbcl.get_incremental import incrementalData
from models.malmo.malmo_run import run
from datetime import datetime
import pandas as pd
import numpy as np
import random, pickle, time, os


def trial(q, pack):  

    #unpack
    pFileNo     = pack[0]
    pMod        = pack[1]   
    pSeed       = pack[2]
    pDataName   = pack[3]
    pLearner    = pack[4]   
    pBias       = pack[5]

    #model parameters
    pNetType        = 'resnet34'            #CNN type
    pNetFit         = 'imagenet'            #dataset for CNN training
    pDistMetric     = 'euclidean'           #distance metric
    pCentroidPred   = 1                     #no. centroids used in ranked voting scheme
    
    #model random
    np.random.seed(pSeed)
    random.seed(pSeed)
    
    #read visual features
    readFile = './utils/features/'+ pDataName + '_' + pNetType + '_' + pNetFit + '_'
    with open(readFile + 'train_features.data', 'rb') as fh: trainFeatRGB   = pickle.load(fh)
    with open(readFile + 'test_features.data', 'rb') as fh:  testFeatRGB    = pickle.load(fh)
    with open(readFile + 'train_labels.data', 'rb') as fh:   trainLabelRGB  = pickle.load(fh)
    with open(readFile + 'test_labels.data', 'rb') as fh:    testLabelRGB   = pickle.load(fh)

    nClassTotal = len(set(testLabelRGB))
    pShotStart  = nClassTotal
    if pDataName == 'grocery':          pDistLim = 13
    elif pDataName == 'cifar':          pDistLim = 17 
    else:                               pDistLim = 15
        
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
        
    #initialize
    centClass       = [[] for x in range(nClassTotal)]      #centroids per class
    centWtClass     = [[] for x in range(nClassTotal)]      #centroid wt per class
    centStdClass    = [[] for x in range(nClassTotal)]      #centroid std per class
    covaClass       = [[] for x in range(nClassTotal)]      #covariance matrices per class
    nShotClass      = [0 for x in range(nClassTotal)]       #image count per class
    weightClass     = [0 for x in range(nClassTotal)]       #weight per class
    rAccClass       = [0 for x in range(nClassTotal)]       #test accuracy per class
    redClass        = [0 for x in range(nClassTotal)]       #redistrict by class
    xTrainBatch     = []                                    #features for SVM case                            
    yTrainBatch     = []                                    #labels for SVM case
    prevSplit       = []                                    #only for redistricting

    #initial examples
    biasTemp        = [random.randint(1,10) for x in range(nClassTotal)]
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
    
        #make new predictions
        rAccClass0 = rAccClass.copy()
        pack = [xTestTot, yTestTot, centClass, pCentroidPred, nClassTotal, weightClass, pDistMetric, covaClass, centWtClass]
        if pLearner == 'CBCLWVS':      rAcc, rAccClass = CBCL_WVS(pack)
        elif pLearner == 'CBCLSVM':    rAcc, rAccClass = CBCL_SVM(pack)
        else:                          rAcc, rAccClass = CBCL_PR(pack)
        
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
                            
        
        #learn and make predictions
        if pBias == 'redistrict':
            rAccClass0 = rAccClass.copy()
            rAcc, rAccClass, prevSplit, redClass = SVM_redistrict(xTrainBatch, yTrainBatch, 
                                                                  xNew, yNew, xTestTot, yTestTot, 
                                                                  prevSplit, redClass, 'SVM', pSeed, 0)
        else:
            rAccClass0 = rAccClass.copy()
            rAcc, rAccClass = SVM_simple(xTrainBatch, yTrainBatch, xTestTot, yTestTot, nClassTotal, 'SVM', pSeed) 


        yTrainCurr = yTrainBatch.copy()
    
    del biasClass, biasTemp
    
    #count observations
    nObsTot         = 0
    nObsTotClass    = [0 for i in range(nClassTotal)]
    for i in range(len(yTrainCurr)): 
        nObsTot +=1
        iClass = yTrainCurr[i]
        nObsTotClass[iClass] +=1
# -----------------------------------------------------------------------------    
    
    #sim parameters
    if pDataName == 'grocery':  pInc = 30
    else: pInc = 60   
    
    pRestock    = True
    
    #initialize sim variables
    iPos            = (0, 0, 0)
    nObsTotClass    = [0 for i in range(nClassTotal)]
    
    #initialize output variables
    final_obs       = [nObsTot]
    final_acc       = [np.round(rAcc,3)]
    final_runTime   = [0]
    final_runDist   = [0]
    final_trainTime = [0]
    runTime         = 0
    runDist         = 0
    trainTime       = 0
    pStatus         = 'IP'
    xLeftover       = []
    yLeftover       = []
    nObsLeftover    = 0
        
    for iInc in range(pInc):
        
        #count available
        nTrainSimClass  = [len(x) for x in xTrainTot]
        nClassEmpty = 0
        for iClass in range(len(nTrainSimClass)):
            nTempClass = nTrainSimClass[iClass]
            if nTempClass == 0: nClassEmpty += 1
        
        #affinity for searching
        if pBias == 'redistrict':   aClass = aff_redistrict(nShotClass, redClass, iInc, pMod)
        else:                       aClass = aff_simple(pBias, centWtClass, centStdClass, rAccClass, rAccClass0, pMod)
            
        #collect images in minecraft simulation
        pack = [pSeed, iPos, aClass, nObsTotClass, nTrainSimClass, 0, pRestock, pDataName, pFileNo, iInc, nShotClass]
        iPos, mcTicks, iDist, iTime, nObsNewClass = run(pack)
        runDist += iDist
        runTime += iTime
        
        #count the new images (assume pRestock True)
        nTrainNewClass = nObsNewClass.copy()
        nObsTotClass = np.array(nObsTotClass)
        nObsTotClass += np.array(nTrainNewClass)
        nObsTotClass = list(nObsTotClass)
        
        if pLearner != 'SVM': #CBCL
            
            train_t0 = np.round(time.time(), 2)

            #process images
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
            
            #make new predictions
            rAccClass0 = rAccClass.copy()
            pack = [xTestTot, yTestTot, centClass, pCentroidPred, nClassTotal, weightClass, pDistMetric, covaClass, centWtClass]
            if pLearner == 'CBCLWVS':      rAcc, rAccClass = CBCL_WVS(pack)
            elif pLearner == 'CBCLSVM':    rAcc, rAccClass = CBCL_SVM(pack)
            else:                          rAcc, rAccClass = CBCL_PR(pack)

            train_t1 = np.round(time.time(), 2)
            addTime = (train_t1 - train_t0)
            trainTime += addTime
        
        else: #SVM
            
            train_t0 = np.round(time.time(), 3)
        
            #process images
            xNew = xLeftover.copy()
            yNew = yLeftover.copy()
            nObsNew = nObsLeftover
            
            for iClass in range(len(nTrainNewClass)):
                for j in range(nTrainNewClass[iClass]):
                    if len(yTrainTot[iClass]) > 0:
                        xTrainObs = xTrainTot[iClass][0]
                        yTrainObs = yTrainTot[iClass][0]
                        xNew.append(xTrainObs)
                        yNew.append(yTrainObs)
                        xTrainBatch.append(xTrainObs)
                        yTrainBatch.append(yTrainObs)
                        del xTrainTot[iClass][0]
                        del yTrainTot[iClass][0]
                        nShotClass[yTrainObs] +=1
                        nObsNew +=1
        
            #learn and make predictions
            if pBias == 'redistrict':
                
                if nObsNew < 10 and nObsLeftover == 0: #too few observations, save what you have and don't train
                    xLeftover = xNew.copy()
                    yLeftover = yNew.copy()
                    nObsLeftover = nObsNew
                
                elif nObsNew < 10 and nObsLeftover != 0: #too few observations (again), save what you have and don't train
                    xLeftover.append(xNew)
                    yLeftover.extend(yNew)
                    nObsLeftover += nObsNew
        
                else: #enough observations, train and test
                    rAccClass0 = rAccClass.copy()
                    rAcc, rAccClass, prevSplit, redClass = SVM_redistrict(xTrainBatch, yTrainBatch, 
                                                                      xNew, yNew, xTestTot, yTestTot, 
                                                                      prevSplit, redClass, 'SVM', pSeed, iInc+1)
                    xLeftover = []
                    yLeftover = []
                    nObsLeftover = 0
            
            else:
                rAccClass0 = rAccClass.copy()
                rAcc, rAccClass = SVM_simple(xTrainBatch, yTrainBatch, xTestTot, yTestTot, nClassTotal, 'SVM', pSeed)   
                
            train_t1 = np.round(time.time(), 3)
            addTime = (train_t1 - train_t0)
            trainTime += addTime
            
        #record results
        if nObsLeftover == 0: nObsTot += nObsNew        
        final_obs.append(nObsTot)
        final_acc.append(np.round(rAcc,3))
        final_runTime.append(np.round(runTime,2))
        final_runDist.append(runDist)
        final_trainTime.append(np.round(trainTime,2))
               
        if iInc == (pInc - 1): pStatus = 'complete'
        output = [pStatus, pFileNo, pMod, pSeed, pDataName, pLearner, pBias, final_obs, final_acc, final_runTime, final_runDist, final_trainTime]   
        if q is not None: q.put(output)
        time.sleep(0.1)  

# -----------------------------------------------------------------------------
            
if __name__ == "__main__":

    #prepare pack for test
    i           = 0
    testPack    = []

    for pMod in [1]: 
        for pSeed in range(10):
            for pDataName in ['grocery', 'cifar100']:
                for pLearner in ['CBCLPR', 'CBCLSVM', 'CBCLWVS']:
                    for pBias in ['classWt', 'uniform', 'clusterWt', 'clusterStdLow', 'clusterStdHigh']:
                        testPack.append([i, pMod, pSeed, pDataName, pLearner, pBias])   
                        i+=1
                for pLearner in ['SVM']:
                    for pBias in ['uniform', 'redistrict']: 
                        testPack.append([i, pMod, pSeed, pDataName, pLearner, pBias])   
                        i+=1 
     
    testPack = testPack[17:33]
    totalResult = [[] for j in range(i)]

    #multi-processing params
    nProcs  = 2
    q       = Queue()
    pHandle = []
    
    #create write path 
    now = datetime.now()
    d0 = now.strftime('%m%d')
    d1 = now.strftime('%Y-%m%d')
    pNetType = 'resnet34'
    FILENAME = './output/{}/{}_{}_{}.xlsx'.format(d0,d1,pNetType, pDataName)
    try:        os.mkdir('./output/{}'.format(d0))
    except:     pass
    
    #run test and write results
    for i in range(nProcs):
        pHandle.append(Process(target=trial, args=(q,testPack[0])))
        testPack.pop(0)
        pHandle[-1].start()
    
    while len(pHandle):
        pHandle = [x for x in pHandle if x.is_alive()]        
        s = nProcs - len(pHandle)
        
        for i in range(s):
            if len(testPack):
                pHandle.append(Process(target=trial, args=(q,testPack[0])))
                testPack.pop(0)
                pHandle[-1].start()
        
        while q.qsize()> 0:
            singleResult = q.get()
            ix = singleResult[1]
            totalResult[ix] = singleResult
            df = pd.DataFrame(totalResult, columns = ['status', 'no.', 'mod', 'seed', 'data', 'learner', 'acs bias', 'observations', 'accuracy', 'run time', 'run distance', 'train time'])
            df.to_excel(FILENAME) 
            
            