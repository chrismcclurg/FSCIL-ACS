# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================
#run in Python 2.7

import json
import models.pepper.pepper as p

# -----------------------------------------------------------------------------
# model parameters
with open('./utils/param.txt') as f: params = f.read()
params          = json.loads(params)
pInc            = params['pInc']

# -----------------------------------------------------------------------------
# MAIN 
p.wakeup()          # initialize
p.test(0, pInc)     # test knowledge

for iInc in range(pInc):
    item_count = 1
    print('\nStarting iteration {} of {} ======================================='.format(iInc+1, pInc) )
    while item_count < 4:
        p.run()                     # search and explore for objects
        result = p.ask(iInc)        # user give examples
        if int(result) > 0:            
            item_count +=1
        p.train()                   # update cluster space
    p.test(iInc+1, pInc)            # test knowledge
     
p.cleanup()         #get rid of temporary files