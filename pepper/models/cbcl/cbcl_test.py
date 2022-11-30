# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================

# imports
from models.cbcl.functions import CBCL_WVS, CBCL_SVM, CBCL_PR, SVM_simple
from models.cbcl.functions import save_var, load_var
import numpy as np
import json
import pandas as pd
from datetime import datetime

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

#update distance
filename = './output/temp_count/cumulative_distance.txt'
f = open(filename, "r") 
data = f.read()
f.close()
with open(filename, 'w') as f:   pass
if data == '':  x = 0.0
else:           x = float(data)
runDist += x

#update run time
filename = './output/temp_count/cumulative_runTime.txt'
f = open(filename, "r") 
data = f.read()
f.close()
with open(filename, 'w') as f:   pass
if data == '':  x = 0.0
else:           x = float(data)
runTime += x

#update train time
filename = './output/temp_count/cumulative_trainTime.txt'
f = open(filename, "r") 
data = f.read()
f.close()
with open(filename, 'w') as f:   pass
if data == '':  x = 0.0
else:           x = float(data)
trainTime += x

#get pred rate
filename = './output/temp_count/guess_true_record.txt'
f = open(filename, "r")
lines = f.read()
lines = lines.split('\n')
lines = [x for x in lines if len(x) > 0]
f.close()

if len(lines) > 0:
    summ = 0
    count = 0
    for line in lines:
        temp = line.split(', ')[-1]
        temp = int(temp)
        print(temp)
        summ  += temp
        count += 1
    iPredRate = float(np.divide(summ,count))

else:
    iPredRate = None


# ----------------------------------------------------------------------------
# make predictions
if pLearner != 'SVM': 
    rAccClass0 = rAccClass.copy()
    pack = [xTestTot, yTestTot, centClass, pCentroidPred, nClassTotal, weightClass, pDistMetric, covaClass, centWtClass]
    if pLearner == 'CBCLWVS':      rAcc, rAccClass, _ = CBCL_WVS(pack)
    elif pLearner == 'CBCLSVM':    rAcc, rAccClass, _ = CBCL_SVM(pack)
    else:                          rAcc, rAccClass, _ = CBCL_PR(pack)
else: 
    rAccClass0 = rAccClass.copy()
    rAcc, rAccClass, _ = SVM_simple(xTrainBatch, yTrainBatch, xTestTot, yTestTot, nClassTotal, 'SVM', pSeed)   

#write accuracy
filepath = './output/temp_count/current_acc.txt'
f = open(filepath, "w")
f.write(str(np.round(rAcc,4)))
f.close()    
          
# record results
final_obs.append(nObsTot)
final_acc.append(np.round(rAcc,3))
final_runTime.append(np.round(runTime,2))
final_runDist.append(runDist)
final_trainTime.append(np.round(trainTime,2))
if not np.isnan(np.float64(iPredRate)): final_predRate.append(np.round(iPredRate,2))
else: final_predRate.append(None)

#clean results
final_completed     = list(range(len(final_obs)))
final_total         = [pInc for x in final_completed]
final_seed          = [pSeed for x in final_completed]
final_data          = [pDataName.lower() for x in final_completed]
final_learner       = [pLearner.lower() for x in final_completed]
final_bias          = [pBias.lower() for x in final_completed]

# write data
data = {'completed': pd.Series(final_completed), 
        'total': pd.Series(final_total),
        'seed': pd.Series(final_seed), 
        'data': pd.Series(final_data), 
        'learner': pd.Series(final_learner), 
        'acs': pd.Series(final_bias), 
        'observations': pd.Series(final_obs), 
        'run time (s)': pd.Series(final_runTime), 
        'run distance (m)': pd.Series(final_runDist), 
        'train time (s)': pd.Series(final_trainTime), 
        'scout accuracy': pd.Series(final_predRate), 
        'test accuracy': pd.Series(final_acc)} 

now = datetime.now()
d = now.strftime('%Y-%m%d')
filename = './output/final/{}_{}_{}-seed_{}.xlsx'.format(pLearner.lower(), pBias, pSeed, d)
df = pd.DataFrame(data)
df.to_excel(filename) 

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
        

            
            