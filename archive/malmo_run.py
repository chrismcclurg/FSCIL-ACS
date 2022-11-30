# =============================================================================
# Incremental Learning (CBCL) with Active Class Selection
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer
# =============================================================================
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a  copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the  rights 
# to use, copy, modify, merge, publish,  distribute,  sublicense,  and/or  sell 
# copies of the Software, and  to  permit  persons  to  whom  the  Software  is 
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall  be  included  in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY  KIND,  EXPRESS  OR 
# IMPLIED, INCLUDING BUT NOT LIMITED  TO  THE  WARRANTIES  OF  MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT  SHALL  THE 
# AUTHORS OR COPYRIGHT HOLDERS BE  LIABLE  FOR  ANY  CLAIM,  DAMAGES  OR  OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
# =============================================================================
from __future__ import print_function
from builtins import range
from malmo import MalmoPython as MP
import sys
import time
from malmo.malmo_functions import get_missionXML, rest, observe
from malmo.malmo_functions import obs_to_path, path_to_move, flag_to_train
from malmo.malmo_functions import flag_to_scatter, check_minima, get_truth
from malmo.malmo_functions import iPos_correct, pos_correct, grass_correct, tp_home
import numpy as np

def run(pack):

    #unpack inputs
    pSeed           = pack[0]
    iPos            = pack[1]
    aClass          = pack[2].copy()
    nObsTotClass    = pack[3].copy()
    nTrainSimClass  = pack[4].copy()
    pObsPerInc      = pack[5]
    pRestock        = pack[6]
    pDataName       = pack[7]
    pFileNo         = pack[8]
    iInc            = pack[9]
    
    #restock stores
    if pRestock: nObsTotClass = [0 for x in nObsTotClass]

    #make sure the initial position is on the map and on a mid-block
    iPos = grass_correct(iPos)    
    iPos = iPos_correct(iPos)
    
    #vars that are initially passed
    iObs        = []                        #raw observations in time step
    prevPos     = iPos                      #previous absolute location  
    nObsCurr    = 0                         #count of examples between centroid update
    iTime       = 0
    iDist       = 0
    runTime     = 0
    
    #flags for movement
    trainFlag   = (-1, -1)                  #flag to stop and collect images
    homeFlag    = False                     #flag to go home, ie set point
    scatFlag    = (False, None, None)       #flag to scatter (Bool, PrevDir, CurrDir)
    sameFlag    = (iPos[0], iPos[1], 1)                 #flag to prevent getting stuck (x, z, nStuck)
    
    #robot initialize
    agent = MP.AgentHost()
    try:
        agent.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent.getUsage())
        exit(1)  
    if agent.receivedArgument("help"):
        print(agent.getUsage())
        exit(0)
    
    my_client_pool = MP.ClientPool()  
    for i in range(10000, 11000): 
        my_client_pool.add(MP.ClientInfo("127.0.0.1", i))
        
        
    #world initialize
    missionXML = get_missionXML(iPos, pSeed, pDataName, pFileNo, iInc) 
    my_mission = MP.MissionSpec(missionXML, True)
    my_mission_record = MP.MissionRecordSpec()
    nAvailClass = get_truth(pSeed, pDataName)
    
    #server initialize
    max_retries = 12
    for retry in range(max_retries):
        try:
            agent.startMission(my_mission, my_client_pool, my_mission_record, 0, "")
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)
    del i, retry, max_retries, missionXML
    
    #loading test
    state = agent.getWorldState()
    while not state.has_mission_begun:
        time.sleep(0.1)
        state = agent.getWorldState()
        for error in state.errors: print("Error:",error.text)
    
    
    #running test
    run_t0 = np.round(time.time(), 2)
    while state.is_mission_running:
        
        while len(state.observations) == 0: 
            rest(0.1, agent)
            state = agent.getWorldState()   
             
        state, iPos, iObs, iTime, iDist = observe(agent, state, iPos, iObs, iTime, iDist, nObsTotClass, nAvailClass, nTrainSimClass, pDataName)
        iPos                      = pos_correct(agent, state, iPos)
        scatFlag, homeFlag        = flag_to_scatter(agent, state, iPos, scatFlag, homeFlag)
        path, trainFlag, homeFlag = obs_to_path(agent, state, iObs, iPos, aClass, scatFlag, homeFlag, pDataName)
        nObsTotClass, nObsCurr    = flag_to_train(agent, state, nObsTotClass, nObsCurr, pObsPerInc, trainFlag)
        prevPos                   = path_to_move(agent, state, path, iPos, prevPos)
        sameFlag, homeFlag        = check_minima(agent, state, iPos, sameFlag, homeFlag)
        
        run_t1 = np.round(time.time(), 2)
        runTime = (run_t1 - run_t0)
        if runTime >= 120: agent.sendCommand("quit")
        
        if sameFlag[2] > 50:   iPos = tp_home(agent, state, iPos) #agent got stuck, so teleport home

        
    return iPos, iTime, iDist, runTime, nObsTotClass

if __name__ == '__main__':

    p1 = 0
    p2 = (0, 0, 0)  
    p3 = [-10, -25, -25, -25, -25, -10, -50, -10, -10, -10, -50, -25, -10, -10, -10, -25, -10, -25, -50, -50, -50, -25, -50, -10, -25, -10, -50, -10, -10, -10, -10, -25, -25, -50, -10, -10, -50, -10, -50, -25, -10, -25, -50, -10, -50, -50, -10, -10, -10, -10, -10, -10, -50, -50, -25, -10, -10, -10, -50, -10, -50, -10, -10, -50, -25, -50, -50, -50, -10, -10, -10, -10, -50, -25, -50, -25, -25, -50, -10, -10, -25, -10, -10, -10, -10, -10, -25, -10, -50, -10, -25, -10, -25, -10, -10, -10, -10, -25, -25, -25]  
    p4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p5 = [10 for i in range(len(p4))]    
    p6 = 25
    p7 = True
    p8 = 'GROCERY'
    
    p3 = [20 if x == -10 else x for x in p3]
    p3 = [-10 if x == -25 else x for x in p3]
    p3 = [-10  if x == -50 else x for x in p3]
    pack = [p1, p2, p3, p4, p5, p6, p7, p8]

    iPos, iTime, iDist, iTime2, nObsTotClass1 = run(pack)
    
    # for iInc in range(10):
    #     pack = [p1, p2, p3, p4, p5]
    #     p2, p4 = run(pack)
    #     print(iInc, p2, p4)
