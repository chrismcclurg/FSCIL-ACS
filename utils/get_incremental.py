# =============================================================================
# Incremental Learning (CBCL) with Active Class Selection
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer
# =============================================================================

import numpy as np
from copy import deepcopy
import random

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
               
        
