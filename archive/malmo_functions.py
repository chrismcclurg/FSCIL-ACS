# =============================================================================
# Incremental Learning (CBCL) with Active Class Selection
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer
# =============================================================================
# MALMO NOTICE
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
from __future__ import division
from builtins import range
import numpy as np
import pandas as pd
import random
import time
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_parent_dir(directory):
    return os.path.dirname(directory)

PARENT_PATH = get_parent_dir(os.getcwd())

def write_DrawItem(xTemp, yGround, zTemp, itemTemp):
    return '\t\t\t\t<DrawItem x="' + str(xTemp) + '" y="' + str(yGround) + '" z="' + str(zTemp) + '" type="' + itemTemp + '"/>\n'

def write_DrawBlock(xTemp, yGround, zTemp, blockTemp):
    return '\t\t\t\t<DrawBlock x="' + str(xTemp) + '" y="' + str(yGround) + '" z="' + str(zTemp) + '" type="' + blockTemp + '"/>\n'

def write_Placement(xTemp, zTemp, yawTemp):
    return '\t\t<Placement x="' + str(xTemp) + '" y="' + str(5) + '" z="' + str(zTemp) + '" yaw="' + str(yawTemp)  + '"/>\n'

def write_Summary(pFileNo, iInc):
    return 'TEST (Trial {}, Iter {})\n'.format(pFileNo, iInc)

def volume(id_block, xmin, xmax, ymin, ymax, zmin, zmax):   
    ans = ''
    for i in range(xmin, xmax+1):
        for j in range(ymin, ymax+1):
            for k in range(zmin, zmax+1):
                ans += write_DrawBlock(i, j, k, id_block) 
    return ans

def building(bldg_mat, xcen, zcen, pSeed, pDataName, bldg_no):
    data = pd.read_excel('utils/env/layout.xlsx')
    data = np.array(data)
    dx = len(data)
    xmin = xcen - (dx/2 +0.5)
    zmin = zcen - (dx/2 +0.5)
    bldg_height = 9
    bldg_floor = 'stonebrick'
    item_height = 4
    
    src = 'utils/env/{}-mapping.xlsx'.format(pDataName)
    data2 = pd.read_excel(src)
    nClassTotal = len(set(list(data2.classNo)))
    data2 = data2[data2.building == bldg_no]
    block = list(data2.block)
    block.sort()
    locs = list(np.arange(0,30, 1))
    freqs = list(np.arange(5,10, 1))
    
    for i in range(len(block), len(locs)):
        block.append('air')

    random.seed(pSeed)
    locBlock = []
    freqBlock = []
    for i in range(len(locs)):
        blockTemp = random.choice(block)
        freqTemp = random.choice(freqs)
        locBlock.append(blockTemp)
        freqBlock.append(freqTemp)
        ix = block.index(blockTemp)
        del block[ix]
        
    #extract count available per increment
    nAvailClass = [0 for x in range(nClassTotal)]
    blocks = list(data2.block)   
    classes = list(data2.classNo)
    for i in range(len(freqBlock)):
        tempBlock   = locBlock[i]
        if tempBlock != 'air':
            tempFreq    = freqBlock[i]
            ix = blocks.index(tempBlock)
            tempClass = classes[ix]
            nAvailClass[tempClass] = tempFreq
        
    count = [0 for i in range(len(locs))]
    spacer = [0 for x in range(10)]
    count = spacer + count
    freqBlock = spacer + freqBlock
    locBlock = spacer + locBlock
    
    ans = ''
    for i in range(len(data)):
        for j in range(len(data[0])):
            val = data[i][j]
            x = int(xmin + i)
            z = int(zmin + j)
            
            if val == 0:    
                ans += volume(bldg_floor, x, x, 1, 3, z, z)

            elif val == 1:  
                ans += volume(bldg_mat, x, x, 1, bldg_height, z, z)
                
            elif val == 3:  
                ans += volume(bldg_mat, x, x, 1, 4, z, z)
                ans += volume('glass_pane', x, x, 5, 7, z, z)
                ans += volume(bldg_mat, x, x, 8, bldg_height, z, z)

            elif val ==4:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)                
                ans += volume(bldg_mat, x, x, 8, bldg_height, z, z)
            
            elif val == 10:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[10], x, x, item_height, item_height, z, z)
                count[10] += 1
                    
            elif val ==11:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[11], x, x, item_height, item_height, z, z)
                count[11] += 1
                    
            elif val ==12:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[12], x, x, item_height, item_height, z, z)
                count[12] += 1

            elif val ==13:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[13], x, x, item_height, item_height, z, z)
                count[13] += 1
       
            elif val ==14:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[14], x, x, item_height, item_height, z, z)
                count[14] += 1
                    
            elif val ==15:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[15], x, x, item_height, item_height, z, z)
                count[15] += 1
                    
            elif val ==16:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[16], x, x, item_height, item_height, z, z)
                count[16] += 1
                    
            elif val ==17:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[17], x, x, item_height, item_height, z, z)
                count[17] += 1
                    
            elif val ==18:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[18], x, x, item_height, item_height, z, z)
                count[18] += 1
                    
            elif val ==19:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[19], x, x, item_height, item_height, z, z)
                count[19] += 1
                    
            elif val ==20:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[20], x, x, item_height, item_height, z, z)
                count[20] += 1
                    
            elif val ==21:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[21], x, x, item_height, item_height, z, z)
                count[21] += 1
                    
            elif val ==22:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[22], x, x, item_height, item_height, z, z)
                count[22] += 1
                    
            elif val ==23:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[23], x, x, item_height, item_height, z, z)
                count[23] += 1
                    
            elif val ==24:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[24], x, x, item_height, item_height, z, z)
                count[24] += 1
                    
            elif val ==25:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[25], x, x, item_height, item_height, z, z)
                count[25] += 1
                    
            elif val ==26:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[26], x, x, item_height, item_height, z, z)
                count[26] += 1
                    
            elif val ==27:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[27], x, x, item_height, item_height, z, z)   
                count[27] += 1
                    
            elif val ==28:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[28], x, x, item_height, item_height, z, z)  
                count[28] += 1
                    
            elif val ==29:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[29], x, x, item_height, item_height, z, z)
                count[29] += 1
            
            elif val ==30:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[30], x, x, item_height, item_height, z, z)
                count[30] += 1
            
            elif val ==31:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[31], x, x, item_height, item_height, z, z)
                count[31] += 1
            
            elif val ==32:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[32], x, x, item_height, item_height, z, z)
                count[32] += 1
            
            elif val ==33:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[33], x, x, item_height, item_height, z, z)
                count[33] += 1
            
            elif val ==34:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[34], x, x, item_height, item_height, z, z)
                count[34] += 1
            
            elif val ==35:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[35], x, x, item_height, item_height, z, z)
                count[35] += 1
            
            elif val ==36:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[36], x, x, item_height, item_height, z, z)
                count[36] += 1
            
            elif val ==37:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[37], x, x, item_height, item_height, z, z)
                count[37] += 1
            
            elif val ==38:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[38], x, x, item_height, item_height, z, z)
                count[38] += 1
            
            elif val ==39:   
                ans += volume(bldg_floor, x, x, 1, 3, z, z)
                ans += volume(locBlock[39], x, x, item_height, item_height, z, z)
                count[39] += 1
            
    return ans, nAvailClass

def get_truth(pSeed, pDataName):
    mat1    = 'brick_block'
    mat2    = 'sandstone'
    mat3    ='cobblestone'
    mat4    = 'log'
    a0      = np.array(building(mat1, 0, 50, pSeed, pDataName, 0)[1])
    a1      = np.array(building(mat2, 0, 50, pSeed, pDataName, 1)[1])
    a2      = np.array(building(mat3, 0, 50, pSeed, pDataName, 2)[1])
    a3      = np.array(building(mat4, 0, 50, pSeed, pDataName, 3)[1])
    ans     = list(a0 + a1 + a2 + a3)
    return ans

def get_missionXML(iPos, pSeed, pDataName, pFileNo, iInc):
    mat1 = 'brick_block'
    mat2 = 'sandstone'
    mat3 ='cobblestone'
    mat4 = 'log'
    x0 = iPos[0]
    z0 = iPos[1]
    yaw0 = iPos[2]
    ans ='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                
                  <About>
                    <Summary>
                    ''' + write_Summary(pFileNo, iInc) + '''
                    </Summary>
                  </About>
                  
                <ServerSection>
                  <ServerInitialConditions>
                    <Time>
                        <StartTime>300</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                  </ServerInitialConditions>
                  <ServerHandlers>
                      <FlatWorldGenerator generatorString="2;7,2x3,2;1,biome_1"/>
                      <DrawingDecorator>     
                          ''' + building(mat1, 0, 50, pSeed, pDataName, 0)[0] + '''
                          ''' + volume(mat1,-21,-21,4,8,47,51) + '''
                          ''' + volume(mat1,19,19,4,8,47,51) + '''
                          ''' + volume(mat1,-3,1,4,8,69,69) + '''
                          
                          ''' + building(mat2, 0, -50, pSeed, pDataName, 1)[0] + '''
                          ''' + volume(mat2,-21,-21,4,8,-53,-49) + '''
                          ''' + volume(mat2,19,19,4,8,-53,-49) + '''
                          ''' + volume(mat2,-3,1,4,8,-71,-71) + '''
                          
                          ''' + building(mat3, 50, 0, pSeed, pDataName, 2)[0] + '''
                          ''' + volume(mat3,47,51,4,8,-21,-21) + '''
                          ''' + volume(mat3,47,51,4,8,19,19) + '''
                          ''' + volume(mat3,69,69,4,8,-3,1) + '''
                          
                          
                          ''' + building(mat4, -50, 0, pSeed, pDataName, 3)[0] + '''
                          ''' + volume(mat4,-53,-49,4,8,-21,-21) + '''
                          ''' + volume(mat4,-53,-49,4,8,19,19) + '''
                          ''' + volume(mat4,-71,-71,4,8,-3,1) + '''
                          
                          ''' + volume('stone',-30,28,3,3,-3,1) + ''' 
                          ''' + volume('stone',-3,1,3,3,-30,28) + ''' 
                          ''' + volume('fence',2,28,4,5,-4,-4) + ''' 
                          ''' + volume('fence',2,28,4,5,2,2) + ''' 
                          ''' + volume('fence',-30,-4,4,5,-4,-4) + ''' 
                          ''' + volume('fence',-30,-4,4,5,2,2) + ''' 
                          ''' + volume('fence',2,2,4,5,2,28) + ''' 
                          ''' + volume('fence',-4,-4,4,5,2,28) + ''' 
                          ''' + volume('fence',2,2,4,5,-30,-4) + ''' 
                          ''' + volume('fence',-4,-4,4,5,-30,-4) + '''
                          
                          ''' + volume('sandstone',-3,1,3,3,-3,1) + ''' 
                          ''' + volume('brick_block',-2,-2,3,3,1,1) + ''' 
                          ''' + volume('brick_block',0,0,3,3,1,1) + ''' 
                          ''' + volume('brick_block',-3,-3,3,3,0,0) + ''' 
                          ''' + volume('brick_block',-1,-1,3,3,0,0) + '''                           
                          ''' + volume('brick_block',1,1,3,3,0,0) + '''                           
                          ''' + volume('brick_block',-2,-2,3,3,-1,-1) + ''' 
                          ''' + volume('brick_block',0,0,3,3,-1,-1) + ''' 
                          ''' + volume('brick_block',-3,-3,3,3,-2,-2) + ''' 
                          ''' + volume('brick_block',-1,-1,3,3,-2,-2) + '''                           
                          ''' + volume('brick_block',1,1,3,3,-2,-2) + '''                              
                          ''' + volume('brick_block',-2,-2,3,3,-3,-3) + ''' 
                          ''' + volume('brick_block',0,0,3,3,-3,-3) + ''' 
                          
                      </DrawingDecorator>
                    </ServerHandlers>
                  </ServerSection>
                  <AgentSection mode="Creative">
                    <Name>ROBOT</Name>
                    <AgentStart>
                        ''' + write_Placement(x0, z0, yaw0) + '''                    
                    </AgentStart>
                    <AgentHandlers>
                      <ObservationFromFullStats/>
                      <ObservationFromNearbyEntities>
                          <Range name="item_obs" xrange="3" yrange="1" zrange="3"/>
                      </ObservationFromNearbyEntities> 
                      <ObservationFromGrid>
                          <Grid name="block_obs">
                              <min x="-15" y="-2" z="-15"/>
                              <max x="15" y="1" z="15"/>
                          </Grid>
                      </ObservationFromGrid>                      
                      <DiscreteMovementCommands/>
                      <MissionQuitCommands/>
                    </AgentHandlers>
                  </AgentSection>
                </Mission>'''
    return ans
    
def rest(t, agent_host):
    agent_host.sendCommand("jump 0")
    agent_host.sendCommand("move 0")
    time.sleep(t)   
    return

def observe(agent_host, state, iPos, iObs, iTime, iDist, nObsTotClass, nAvailClass, nTrainSimClass, pDataName):
    trained = nObsTotClass.copy()   #trained count
    availInc = nAvailClass.copy()  #available in an increment
    availTot = nTrainSimClass.copy() #available through test
    
    new_state = agent_host.getWorldState()
    if new_state.is_mission_running:
        if new_state.number_of_observations_since_last_state > 0: state = new_state
        msg = state.observations[-1].text
        obs = json.loads(msg)   
        x = np.round(obs.get(u'XPos', 0))
        z = np.round(obs.get(u'ZPos', 0))
        yaw = np.round(obs.get(u'Yaw', 0)) 
        iObs = obs.get(u'block_obs', 0)
        iTime = obs.get(u'TimeAlive', 0) 
        iDist = obs.get(u'DistanceTravelled', 0) 
        iPos = (x, z, yaw)
        
        #read the mapping from observation to dataset
        src = 'utils/env/{}-mapping.xlsx'.format(pDataName)
        mapping = pd.read_excel(src)
        blocks = list(mapping.block)
        classno = list(mapping.classNo)
                
        #remove items exceeding available
        counts = [0 for i in range(len(availInc))]
        if type(iObs) == int: iObs = []
        for i in range(len(iObs)): 
            temp = iObs[i]
            if temp in blocks:
                ix = blocks.index(temp)
                tempClass = classno[ix]
                if counts[tempClass] == availInc[tempClass] or counts[tempClass] == availTot[tempClass]:
                    iObs[i] = 'air'
                else: counts[tempClass] += 1
        
        #remove already trained
        for i in range(len(iObs)): 
            temp = iObs[i]
            if temp in blocks:
                ix = blocks.index(temp)
                tempClass = classno[ix]
                
                if trained[tempClass] > 0:
                    iObs[i] = 'air'
                    trained[tempClass] -= 1

    else: 
        state = new_state
    return state, iPos, iObs, iTime, iDist


def obs_to_path(agent_host, state, iObs, iPos, xTrainWts, scatFlag, homeFlag, pDataName):
    
    trainFlag = (-1, -1)
    if state.is_mission_running:
    
        #re-format the MC observations from single arrauy
        i = 0
        k = 0
        collect = [[[] for x in range(31)] for z in range(31)]
        for temp in iObs:
            collect[i][k].append(temp)
            i+=1
            if i > 30:
                k+= 1
                i = 0
                if k > 30: k = 0
        
        #read the mapping from observation to dataset
        src = 'utils/env/{}-mapping.xlsx'.format(pDataName)
        mapping = pd.read_excel(src)        
        blocks = list(mapping.block)
        classno = list(mapping.classNo)
        
        #make a matrix of classno per x-z location
        obs = [[] for x in range(31)]
        for i in range(len(collect)):
            for k in range(len(collect[0])):
                temp = collect[k][i]
                temp = [x for x in temp if x in blocks]
                temp = [classno[blocks.index(x)] for x in temp]
                if len(temp)>0: temp2 = int(temp[0])
                else: temp2 = -1
                obs[i].append(temp2)
        obs = np.matrix(obs)
        
        #make an orientation arrow for plotting
        x0 = int(np.round(len(obs)/2,0))-1
        z0 = int(np.round(len(obs)/2,0))-1
        yaw = iPos[2]
        #print(yaw)
        if yaw >= 45 and yaw <= 135: #W (90)
            dx = -1
            dz = 0
        elif (yaw >=135 and yaw <= 180) or yaw < -135:  #N (180)
            dx = 0 
            dz = -1
        elif (yaw <= -45 and yaw >= -135) or (yaw >=225 and yaw <= 315):  #E (-90 or 270)
            dx = 1 
            dz = 0
        else:   #S (0)
            dx = 0
            dz = 1
        
        #make dots for plotting    
        xdots = np.arange(0, 31, 3)
        zdots = np.arange(0, 31, 3) 
        x1 = []
        z1 = []
        for i in xdots: 
            for k in zdots:
                x1.append(i)
                z1.append(k)
        
        #process points of attraction
        nskip = 1
        obsFilt = obs[::nskip, ::nskip]
        x = np.arange(-15,16,nskip)
        z = np.arange(-15,16,nskip) 
        X,Z = np.meshgrid(x,z)
        u = np.zeros_like(X)
        v = np.zeros_like(Z)
        xvec = []
        zvec = []
        avec = []
        considered = []
        for i in range(len(obsFilt)):
            for j in range(len(obsFilt)):
                if obsFilt[i,j] > -1:
                    if obsFilt[i,j] not in considered:
                        xvec.append(X[i,j])
                        zvec.append(Z[i,j])
                        avec.append(xTrainWts[obsFilt[i,j]])
                        considered.append(obsFilt[i,j])
                        
        #flag if in position to learn
        #print(obsFilt[15,15])
        if obsFilt[15,15] > 0 and not scatFlag[0] and not homeFlag:
            trainClass = obsFilt[15,15] 
            #print('learning class {}'.format(trainClass))
            nTrain = 0
            for i in range(-2, 3):
                for j in range(-2, 3):
                    temp = obsFilt[15 + i, 15 + j]
                    if temp == trainClass: nTrain +=1
            trainFlag = (trainClass, nTrain)
        #print(trainFlag)
        
        #create velocity field
        for k in range(len(xvec)):
            xi = xvec[k]
            zi = zvec[k]
            ai = avec[k]
            for i in range(len(x)):
                for j in range(len(z)):
                    d = np.sqrt((X[i,j] - xi)**2 + (Z[i,j]-zi)**2)    
                    theta = np.arctan2((Z[i,j]-zi),(X[i,j] - xi)) 
                    if d<1:
                        u[i][j] = 0
                        v[i][j] = 0
                    else: 
                        u[i][j] += ai*np.cos(theta)/d
                        v[i][j] += ai*np.sin(theta)/d
        
        #plot 1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
        plt.style.use('seaborn-whitegrid')
        sns.heatmap(obs, vmin = 0, vmax = 100, xticklabels = False, yticklabels = False, ax = ax1, cbar = False)
        ax1.arrow(x0, z0, dx, dz, width = 0.1, head_width = 0.5, fc = 'white', ec = 'white', zorder = 3)
        ax1.scatter(x1, z1, c = 'white', s = 2)
            
        #check if any attraction
        ubar = u[15,15]
        vbar = v[15,15]
        fMove = (np.abs(ubar) > 0) or (np.abs(vbar) > 0)
        fHome = homeFlag
        fScat = scatFlag[0]
        fTrain = (trainFlag[0] != -1)
        #print(ubar, vbar, fMove)
        
        #if not fMove and not fTrain and not fScat: fHome = True
        
        # Check different cases of agent status (home, scatter, move, train)
        # ---------------------------------------------------------------------                        
        if fScat:  #update the velocity field with scatter direction
            print('SCATTER')
            x = np.arange(-15,16,nskip)
            z = np.arange(-15,16,nskip) 
            X,Z = np.meshgrid(x,z)
            currDir = scatFlag[2]
            if currDir == 'N':
                uS = 0
                vS = -1
            elif currDir == 'S':
                uS = 0
                vS = 1
            elif currDir == 'E':
                uS = 1
                vS = 0
            elif currDir == 'W':
                uS = -1
                vS = 0
                
            u = uS * np.ones_like(X)
            v = vS * np.ones_like(Z)
            
        # ---------------------------------------------------------------------                               
        elif fHome: # update velocity field with home direction
            print('HOME')
            u = np.zeros_like(X)
            v = np.zeros_like(Z)
            x0 = iPos[0]
            z0 = iPos[1]
            dz = -0.5 - z0
            dx = -0.5 - x0
            if np.abs(x0) > np.abs(z0):     #E/W buildings (center Z, then X)
                if np.abs(dz) > 1: 
                    for i in range(len(x)):
                        for j in range(len(z)):
                            u[i][j] += 0
                            v[i][j] += dz/np.abs(dz)
                else: 
                    for i in range(len(x)):
                        for j in range(len(z)):
                            u[i][j] += dx/np.abs(dx)
                            v[i][j] += 0
            
            else:                           #N/S buildings (center X, then Z)
                if np.abs(dx) > 1: 
                    for i in range(len(x)):
                        for j in range(len(z)):
                            u[i][j] += dx/np.abs(dx)
                            v[i][j] += 0
                
                else: 
                    for i in range(len(x)):
                        for j in range(len(z)):
                            u[i][j] += 0
                            v[i][j] += dz/np.abs(dz)
        elif fTrain:
            print('TRAIN')
            #print(trainFlag)
            
        # ---------------------------------------------------------------------                               
        elif not fMove: # try to get out of the minima
            print('SCRAMBLE')
            u[15,15] = random.choice([-1, 1])
            v[15,15] = random.choice([-1, 1])

        else:
            print('MOVE')
        # ---------------------------------------------------------------------     
        
        #plot 2
        ax2.quiver(X, Z, u, v)
        ax2.xaxis.set_ticks([])
        ax2.yaxis.set_ticks([])
        ax2.invert_yaxis()
        stream = ax2.streamplot(X,z,u,v, start_points= np.array([[0,0]]), integration_direction ='forward')               
        segments = stream.lines.get_segments()
        
        # stream = plt.streamplot(X,z,u,v, start_points= np.array([[0,0]]), integration_direction ='forward')               
        # segments = stream.lines.get_segments()
        
        ubar = u[15,15]
        vbar = v[15,15]
        print(ubar, vbar)  

        if len(segments) == 0: #manually make a segment
            if np.abs(vbar) > np.abs(ubar):
                if vbar > 0: segments = [[tuple([0, 0]), tuple([0, 1])], [tuple([0, 1]), tuple([0, 2])]]
                if vbar < 0: segments = [[tuple([0, 0]), tuple([0, -1])], [tuple([0, -1]), tuple([0, -2])]]
            else: 
                if ubar > 0: segments = [[tuple([0, 0]), tuple([1, 0])], [tuple([1, 0]), tuple([2, 0])]]
                if ubar < 0: segments = [[tuple([0, 0]), tuple([-1, 0])], [tuple([-1, 0]), tuple([-2, 0])]]
          
        #make discretized path
        path = [tuple([0, 0])]
        for i in range(len(segments)):
            temp = tuple(np.round(segments[i][1], 0))
            temp = tuple([int(x) for x in temp])
            if temp[0] != path[-1][0] or temp[1] != path[-1][1]: path.append(temp)      
        
        #overlay the path on plot 2
        if fMove:
            xPath = []
            yPath = []
            for i in range(len(path)):
                xPath.append(path[i][0])
                yPath.append(path[i][1])
            ax2.plot(xPath, yPath, 'purple')
            ax2.add_patch(plt.Circle((0, 0), 0.5))
            
            #plot points of attraction
            if not fHome and not fScat:
                xPos = []
                zPos = []
                xNeg = []
                zNeg = []
                for i in range(len(avec)):
                    ai = avec[i]
                    if ai > 0:
                        xPos.append(xvec[i])
                        zPos.append(zvec[i])
                    else:
                        xNeg.append(xvec[i])
                        zNeg.append(zvec[i]) 
                ax2.scatter(xPos, zPos, s= 10, c = 'red')
                ax2.scatter(xNeg, zNeg, s= 10, c = 'green')
        
        plt.subplots_adjust(wspace=0.025, hspace=0)
        plt.show()
        
        homeFlag = fHome
    else:  path = [tuple([0, 0])]
    return path, trainFlag, homeFlag

def path_to_move(agent_host, state, path, iPos, prevPos):    
    if state.is_mission_running:
        if len(path) > 1:
            x0 = path[0][0]
            z0 = path[0][1]
            x1 = path[1][0]
            z1 = path[1][1]
            #print(1)
        
        else:
            x0 = prevPos[0]
            z0 = prevPos[1]
            x1 = iPos[0]
            z1 = iPos[1]
            #print(2)
            
        dx = x1 - x0
        dz = z1 - z0
        yaw0 = iPos[2]
        
        if dz > 0: 
            if yaw0 == 90: 
                agent_host.sendCommand("turn -1")
            elif yaw0 == 180: 
                agent_host.sendCommand("turn -1")
                agent_host.sendCommand("turn -1")
            elif yaw0 == -90 or yaw0 == 270: 
                agent_host.sendCommand("turn 1")
                     
        elif dz < 0: 
            if yaw0 == -90 or yaw0 == 270: 
                agent_host.sendCommand("turn -1")
            elif yaw0 == 0: 
                agent_host.sendCommand("turn -1")
                agent_host.sendCommand("turn -1")
            elif yaw0 == 90: 
                agent_host.sendCommand("turn 1")
                
        elif dx > 0: 
            if yaw0 == 0: 
                agent_host.sendCommand("turn -1")
            elif yaw0 == 90: 
                agent_host.sendCommand("turn -1")
                agent_host.sendCommand("turn -1")
            elif yaw0 == 180: 
                agent_host.sendCommand("turn 1")
                
        elif dx < 0: 
            if yaw0 == 180: 
                agent_host.sendCommand("turn -1")
            elif yaw0 == -90 or yaw0 == 270: 
                agent_host.sendCommand("turn -1")
                agent_host.sendCommand("turn -1")
            elif yaw0 == 0: 
                agent_host.sendCommand("turn 1")
        
        rest(0.1, agent_host)
        
        if dz > 0: 
            agent_host.sendCommand("jumpsouth 1")
        elif dz < 0: 
            agent_host.sendCommand("jumpnorth 1")
        elif dx > 0: 
            agent_host.sendCommand("jumpeast 1")
        elif dx < 0: 
            agent_host.sendCommand("jumpwest 1")
        
        #rest(0.1, agent_host)        
        prevPos = (iPos[0], iPos[1], iPos[2])
        
    return prevPos

def flag_to_train(agent, state, nTrainedClass, nObsCurr, pObsPerInc, trainFlag):
    if state.is_mission_running:
        if trainFlag[0] != -1:
            classNew = trainFlag[0]
            nNew = trainFlag[1]
            for i in range(nNew):
                # if nObsCurr < pObsPerInc:
                nTrainedClass[classNew] += 1
                nObsCurr +=1
                print('no. {} for class {}, {} total'.format(i, classNew, nObsCurr))
                rest(0.5, agent)
    return nTrainedClass, nObsCurr

def flag_to_scatter(agent, state, iPos, scatFlag, homeFlag):
    if state.is_mission_running:
        
        x = iPos[0]
        z = iPos[1]
        possibleDirs = ['N', 'E', 'S', 'W']
        
        prevDir = scatFlag[1]
        currDir = scatFlag[2]
        fScat = scatFlag[0]
        
        dist = np.sqrt(x**2 + z**2)
        
        if not fScat and dist < 1:                #initialize scattering
            fScat = True
            homeFlag = False
            if prevDir is not None: 
                ix = possibleDirs.index(prevDir)
                del possibleDirs[ix]
            currDir = random.sample(possibleDirs, 1)[0]
        
        elif fScat and dist < 40:                 #still scattering 
            fScat = True
            
        elif fScat and dist >= 40:                #finished scattering  
            fScat = False
            prevDir = currDir
            
        else: 
            fScat = False                       #do not scatter
        
        scatFlag = (fScat, prevDir, currDir)
        
    return scatFlag, homeFlag

def check_minima(agent, state, iPos, sameFlag, homeFlag):
    if state.is_mission_running:
        tol = 5
        
        x1 = iPos[0]
        z1 = iPos[1]
        
        x0 = sameFlag[0]
        z0 = sameFlag[1]
        nSame = sameFlag[2]
        
        dist = np.sqrt((x1-x0)**2 + (z1-z0)**2)
            
        if dist <= tol: nSame +=1
        else: 
            nSame = 1
            x0 = x1
            z0 = z1
        
        if nSame > 30: 
            homeFlag = True
        
        if nSame > 25:
            print('MINIMA : {}'.format(nSame))

        sameFlag = (x0, z0, nSame) 
    
    return sameFlag, homeFlag   

def iPos_correct(iPos):
    x = iPos[0]
    z = iPos[1]
    yaw = iPos[2]
    
    if (2*x) %2==0: 
        if x<0: x+=0.5
        else: x-=0.5
  
    if (2*z) %2==0:
        if z<0: z+=0.5
        else: z-=0.5
            
    iPos = (x,z,yaw)
    return iPos  

def pos_correct(agent, state, iPos):
    if state.is_mission_running:
        iPos = iPos_correct(iPos)
        x = iPos[0]
        z = iPos[1]
        teleport = 'tp ' +str(x) + ' ' + str(5) + ' ' + str(z)
        agent.sendCommand(teleport)
    return iPos

def grass_correct(iPos):
    
    x = iPos[0]
    z = iPos[1]
    yaw = iPos[2]
    
    xabs = np.abs(x)
    zabs = np.abs(z)
    if xabs < 31 and zabs < 31: #outside of buildings
        if xabs < zabs: x = 0   #put on path
        else: z = 0             #put on path
    iPos = (x,z,yaw)
    return iPos  

def tp_home(agent, state, iPos):
    x = iPos[0]
    z = iPos[1]
    yaw = iPos[2]
    
    if x > 0: newX = 0.5
    else: newX = -0.5
    
    if z > 0: newZ = 0.5
    else: newZ = -0.5
    
    if state.is_mission_running:
        teleport = 'tp ' +str(newX) + ' ' + str(5) + ' ' + str(newZ)
        agent.sendCommand(teleport)
    iPos = (newX, newZ, yaw)
    return iPos
