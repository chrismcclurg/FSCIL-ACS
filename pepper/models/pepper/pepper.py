# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================

#imports
import os
import os.path
import qi
import random
import numpy as np
from PIL import Image
import time, subprocess
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.cm as cm
from env import get_path, get_actions

### ---------------------------------------------------------------------------
# START UP

IP      = '192.168.0.156'
PORT    = '9559'
VOCAL   = False
logger  = qi.logging.Logger("network")
session = qi.Session()
session.connect("tcp://" + IP + ":" + PORT)  
_nav    = session.service("ALNavigation")
_mot    = session.service("ALMotion")
_pos    = session.service("ALRobotPosture")
_loc    = session.service("ALLocalization")
_tts    = session.service("ALTextToSpeech")
_vid    = session.service("ALVideoDevice")
_mem    = session.service("ALMemory")
_bat    = session.service("ALBattery")
_dcm    = session.service("DCM")
_tra    = session.service("ALTracker")
_bas    = session.service("ALBasicAwareness")
logger.info("=> Pepper connected to session.")

### ---------------------------------------------------------------------------
# BASIC

def wakeup():
    _mot.wakeUp()
    _pos.goToPosture("StandInit", 0.1)
    _mot.setOrthogonalSecurityDistance(0.002)
    _mot.setTangentialSecurityDistance(0.002)
    _mot.setExternalCollisionProtectionEnabled("Arms", False)
    t = _dcm.getTime(0)
    _dcm.set(["Device/SubDeviceList/Platform/LaserSensor/Front/Reg/OperationMode/Actuator/Value", "Merge", [[1.0, t]]]) 
    _dcm.set(["Device/SubDeviceList/Platform/LaserSensor/Left/Reg/OperationMode/Actuator/Value", "Merge", [[1.0, t]]])
    logger.info("=> Pepper has turned on sensors.")
    batLev = _bat.getBatteryCharge()
    logger.info("=> Pepper is {}% charged.".format(batLev))
    subprocess.Popen("conda run -n py38 python models/cbcl/cbcl_init.py".split(), stdout=subprocess.PIPE)
    time.sleep(5)
    logger.info("=> Pepper has initialized cluster space. ")    
    return

def get_offsets():
    global xOffset, yOffset, tOffset
    xOffset, yOffset, tOffset = _mot.getRobotPosition(True)

def speak(sentence):
    _tts.say(sentence)
    return

def declare(var):
    _tra.pointAt('Arms', [1, 0, 0], 0, 1)
    sentence = 'I want more examples of {}'.format(var)
    _tts.say(sentence)

def pos():   
    temp = _mot.getRobotPosition(True)
    x_m = temp[0]
    y_m = temp[1]
    t_rad = temp[2]
    x, y, t = SI_to_blocks(x_m, y_m, t_rad)
    return x, y, t
    
def blocks_to_SI(sx, sy, st, blockSize = 0.230):  
    sx_m    = float(sx*blockSize)
    sy_m    = float(sy*blockSize)
    st_rad  = np.radians(st)
    if st_rad > np.pi:      st_rad = float(np.round(-2.0*np.pi + st_rad, 2))
    elif st_rad < -np.pi:   st_rad = float(np.round(2.0*np.pi + st_rad, 2))
    else:                   st_rad = float(np.round(st_rad, 2))
    return sx_m, sy_m, st_rad

def SI_to_blocks(sx_m, sy_m, st_rad, blockSize = 0.230):
    sx = float(np.round(sx_m/blockSize,2))
    sy = float(np.round(sy_m/blockSize,2))
    st  = np.degrees(st_rad)
    if st > 180.:      st = float(np.round(-360. + st, 2))
    elif st < -180.:   st = float(np.round(360. + st, 2))
    else:              st = float(np.round(st, 2))
    return sx, sy, st

def abs_to_rel(uxa, uya, uta, ta):
    uxr = float(np.round( uxa*np.cos(np.radians(ta)) + uya*np.sin(np.radians(ta)),2))
    uyr = float(np.round(-uxa*np.sin(np.radians(ta)) + uya*np.cos(np.radians(ta)),2))   
    utr = float(np.round(uta,2))
    return uxr, uyr, utr

def rel_to_abs(uxr, uyr, utr, ta):
    uxa = float(np.round( uxr*np.cos(np.radians(ta)) - uyr*np.sin(np.radians(ta)),2))
    uya = float(np.round( uxr*np.sin(np.radians(ta)) + uyr*np.cos(np.radians(ta)),2))   
    uta = float(np.round(utr,2))
    return uxa, uya, uta

def end_pos(ux, uy, ut):
    x0, y0, t0 = pos()                                  #initial_pos
    uxa, uya, uta = rel_to_abs(ux, uy, ut, t0)          #absolute move
    xF = x0 + uxa                                       #final pos
    yF = y0 + uya                                       #final pos
    tF = t0 + uta                                       #final pos   
    return xF, yF, tF
    
def rel_error(xF, yF, tF):
    x1, y1, t1 = pos()
    exa = xF - x1
    eya = yF - y1
    eta = tF - t1
    ex, ey, et = abs_to_rel(exa, eya, eta, t1)
    return ex, ey, et

def write_dist(iDist):
    filename = './output/temp_count/cumulative_distance.txt'       
    f = open(filename, "r") 
    data = f.read()
    f.close()
    with open(filename, 'w') as f:   pass
    if data == '':  dist = 0.0
    else:           dist = float(data)
    
    dist += iDist
    outfile = open(filename, "w")
    outfile.writelines(str(dist))
    outfile.close()
    outfile.close()
    return

def write_runTime(iRT):
    filename = './output/temp_count/cumulative_runTime.txt'       
    f = open(filename, "r") 
    data = f.read()
    f.close()
    with open(filename, 'w') as f:   pass
    if data == '':  tm = 0.0
    else:           tm = float(data)
    
    tm += iRT
    outfile = open(filename, "w")
    outfile.writelines(str(tm))
    outfile.close()
    outfile.close()
    return

def write_trainTime(iTT):
    filename = './output/temp_count/cumulative_trainTime.txt'       
    f = open(filename, "r") 
    data = f.read()
    f.close()
    with open(filename, 'w') as f:   pass
    if data == '':  tm = 0.0
    else:           tm = float(data)
    
    tm += iTT
    outfile = open(filename, "w")
    outfile.writelines(str(tm))
    outfile.close()
    outfile.close()
    return
    
### ---------------------------------------------------------------------------
# SENSORY

# quick check for exploring
def quick_check():  
    logger.info('=> Pepper is looking for free space.')
    _bas.pauseAwareness()       
    nInc = 6                       #number of turns
    inc = 2.0*np.pi / nInc         #increment of turn
    theta0 = -inc               #initialize theta0
    xObs = []
    yObs = []
    for ix in range(nInc):
        theta = ix*inc
        xTemp, yTemp = get_sensors(theta)
        xObs.extend(xTemp)
        yObs.extend(yTemp)
        completed = _mot.moveTo(0.0, 0.0, theta - theta0, _mot.getMoveConfig("Max"))
        if not completed:
            speak('Please fix my base so I can continue.')
            raw_input('Fix Pepper base. Press Enter to continue.')
        theta0 = theta
    _bas.resumeAwareness() 
    maze, _, _ = get_maze(xObs, yObs) 
    end_pos = get_freeSpace(maze)
    return maze, None, end_pos

#full check for looking for objects
def check():  
    _bas.pauseAwareness()       
    nInc = 6                       #number of turns
    inc = 2.0*np.pi / nInc         #increment of turn
    theta0 = -inc               #initialize theta0
    xObs = []
    yObs = []
    speak('I am going to scout out this area.')
    for ix in range(nInc):
        theta = ix*inc
        xTemp, yTemp = get_sensors(theta)
        get_visual(theta, ix, nInc)
        xObs.extend(xTemp)
        yObs.extend(yTemp)
        completed = _mot.moveTo(0.0, 0.0, theta - theta0, _mot.getMoveConfig("Max"))
        if not completed:
            speak('Please fix my base so I can continue.')
            raw_input('Fix Pepper base. Press Enter to continue.')
        theta0 = theta
    _bas.resumeAwareness() 
    
    speak('Lets see...')
    logger.info('=> Pepper is localizing objects in images.')     
    subprocess.Popen("conda run -n py38 python models/yolo/object_detector.py".split(), stdout=subprocess.PIPE) 
    
    #wait until count file is non-empty
    data = []
    while len(data) == 0:
        f = open('./output/temp_count/cropped_count.txt')  
        data = f.readlines()
        f.close()
        time.sleep(0.1) 

    #get count, then empty count file
    nCropped = int(data[0])
    with open('./output/temp_count/cropped_count.txt', 'w') as f: pass

    #wait until cropped images are in the folder
    while len(os.listdir('./output/temp_count/images/1_cropped/')) < nCropped: 
        time.sleep(0.1)
    
    #report count of cropped images
    logger.info('=> Pepper located {} objects.'.format(nCropped))     
    speak('I think there are {} objects.'.format(nCropped))
    
    #make predictions on cropped images (only if detected)
    thetas, affs, items = None, None, None
    if nCropped > 0: 
        speak('Let me think about what the objects are..')
        logger.info('=> Pepper is making predictions on cropped images. ')     
        subprocess.Popen("conda run -n py38 python models/cbcl/cbcl_predict.py".split(), stdout=subprocess.PIPE)
    
        #wait until label file is non-empty
        labels = []
        while len(labels) == 0:
            f = open('./output/temp_count/images/2_guesses/labels.txt')  
            labels = f.readlines()
            f.close()
            time.sleep(0.1) 
        
        #empty label file
        with open('./output/temp_count/images/2_guesses/labels.txt', 'w') as f: pass
        
        #localize items
        thetas, affs, items  = get_items(nInc, nCropped, labels)
        logger.info('=> Pepper predicted {} unique items. '.format(len(set(items))))   
 
    #determine best item and path to that item
    maze, best_item, end_pos = get_maze(xObs, yObs, thetas, affs, items) 

    if best_item is not None: 
        CLASSES = './utils/grocery.txt'
        with open(CLASSES, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        iBest = classes.index(best_item)
        logger.info('=> Pepper has determined ({}) {} to be the best option.'.format(iBest, best_item))     
    else: logger.info('=> Pepper did not find a good option.')     
    return maze, best_item, end_pos

def get_sensors(theta):
    xRel_final = []
    yRel_final = []
    for j in range(0,15):
        if j+1<10:
            SEG = "0" + str(j+1)
        else:
            SEG = str(j+1)
        #front
        Y =  (_mem.getData("Device/SubDeviceList/Platform/LaserSensor/Front/Horizontal/Seg{}/X/Sensor/Value".format(SEG)) + 0.05620)
        X = -(_mem.getData("Device/SubDeviceList/Platform/LaserSensor/Front/Horizontal/Seg{}/Y/Sensor/Value".format(SEG)) + 0.00000)
        xRel = np.round(X*np.cos(theta) - Y*np.sin(theta),2)
        yRel = np.round(X*np.sin(theta) + Y*np.cos(theta),2)
        xRel, yRel, _ = SI_to_blocks(xRel, yRel, 0)
        xRel_final.append(np.round(xRel,1))
        yRel_final.append(np.round(yRel,1))
        #left
        X = -(_mem.getData("Device/SubDeviceList/Platform/LaserSensor/Left/Horizontal/Seg{}/X/Sensor/Value".format(SEG)) + 0.08990)              
        Y = -(_mem.getData("Device/SubDeviceList/Platform/LaserSensor/Left/Horizontal/Seg{}/Y/Sensor/Value".format(SEG)) - 0.01800)
        xRel = np.round(X*np.cos(theta) - Y*np.sin(theta),2)
        yRel = np.round(X*np.sin(theta) + Y*np.cos(theta),2)
        xRel, yRel, _ = SI_to_blocks(xRel, yRel, 0)
        xRel_final.append(np.round(xRel,1))
        yRel_final.append(np.round(yRel,1))        
    return xRel_final, yRel_final #output block units

def get_visual(theta, ix, nInc):
    #look in direction
    y_m = 0
    x_m, z_m, _ = blocks_to_SI(2, 1, 0)
    _tra.lookAt([x_m, y_m, z_m], 1, True)
    
    #take image (see documentation for parameters)
    resId           = 3
    camId           = 0 
    imgName         = "{}".format(str(random.randint(0, 999999999999)))
    imgCam          = _vid.subscribeCamera(imgName, camId, resId, 11, 10)
    imgRGB          = _vid.getImageRemote(imgCam)    
    imgWidth        = imgRGB[0]
    imgHeight       = imgRGB[1]
    imgArray        = imgRGB[6]
    imgStr          = str(bytearray(imgArray))
    img             = Image.frombytes("RGB", (imgWidth, imgHeight), imgStr)
    img.save('./output/temp_count/images/0_raw/{}_rgb.png'.format(ix))    
    logger.info("=> Pepper saved image {} of {}.".format(ix+1, nInc))
    return


#NEED TO FIX LENGTH ERROR OF IMAGES, I DON'T THINK THE DATUM HAS SAME LENGTH
def get_items(nInc, nCropped, labels): 
   
   inc          = 2.0*np.pi / nInc
   imageList    = os.listdir('./output/temp_count/images/1_cropped/')
   thetas       = []
   affs         = []
   items        = []
       
   for i in range(nCropped):
       
       #get angle of item
       image = imageList[i]
       tc = inc*int(image.split('_')[0])+ np.pi/2       #theta position of image (rad)
       _, _, tc = SI_to_blocks(0,0,tc)                  #convert to degrees
       dt = int(image.split('_')[1])                    #pertubation from center (percentage)
       teff = np.round(tc - (dt-50)/100.*56.3/2.,3)     #effective angle (deg)
       
       #get item and affinity
       data = labels[i].split('\t')
       
       #append to list
       thetas.append(teff)
       items.append(data[1].strip())
       affs.append(int(data[2].strip()))
       
   return thetas, affs, items

def get_maze(x, y, thetas = None, affs = None, items = None):
    #initialize maze
    maze = np.zeros((21,21))
    xOffset = 10
    yOffset = 10
    
    #obstacles from sensor (get radial coordinates)
    r = []
    t = []
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if np.abs(xi) < 10. and np.abs(yi) < 15.:
            ri = np.round(np.sqrt(xi**2+ yi**2),2)
            ti = np.round(np.arctan2(yi,xi)*180./np.pi,2) #straight ahead is 90
            r.append(ri)
            t.append(ti)

    #rays from sensor (update maze)
    for j in range(-10,11):
        for i in range(-10,11):
            xi = i
            yj = j
            rji = np.round(np.sqrt(xi**2+ yj**2),2)
            tji = np.round(np.arctan2(yj,xi)*180./np.pi,2) #straight ahead is 90
            t_filter = [ti for ti in t if np.abs(tji - ti) < 10.0]
            for ti in t_filter:
                ix = t.index(ti)
                if rji > r[ix]-1:
                    maze[yj+yOffset][xi+xOffset] = 1
    
    #obstacles from sensor (update maze)
    for ix in range(len(x)):
        j = int(np.round(x[ix],0)) + xOffset
        i = int(np.round(y[ix],0)) + yOffset
        if i > len(maze)-1 or i < 0: continue
        if j > len(maze[0])-1 or j < 0: continue
        maze[i][j] = 7
        
    #objects from vision (update maze)
        best_score = 0.
        best_item = None
        end_pos = None
        best_itemNo = None
        
    
    if thetas is not None:
        count = 0
        for ix in range(len(thetas)):
            ti = thetas[ix] 
            dt = np.array([np.abs(ti - tj) for tj in t])
            arg_min = np.argmin(dt)
            rClose = r[arg_min]
            tClose = t[arg_min]*np.pi/180.
            xClose = rClose*np.cos(tClose)
            yClose = rClose*np.sin(tClose)        
            i = int(np.round(xClose,0)) + xOffset
            j = int(np.round(yClose,0)) + yOffset
            
            #constrain to map
            if j < 0: j = 0 
            if j > 20: j = 20
            if i < 0: i = 0
            if i > 20: i = 20
            
            maze[j][i] = count + 10
            count += 1
            score = -affs[ix] / rClose
            if score > best_score:
                best_score = score 
                best_item = items[ix]
                end_pos = (xClose, yClose)
                best_itemNo = ix
            
    if end_pos is not None:
        i = int(np.round(end_pos[0],0)) + xOffset
        j = int(np.round(end_pos[1],0)) + yOffset
        maze[j,i] = 9
    
    if best_itemNo is not None:
        plot_choice(maze, best_itemNo)
    
    return maze, best_item, end_pos

def get_freeSpace(maze, mag = 5):
    temp = maze.copy()
    temp[temp > 1] = 1    
    poss = []
    poss.append(np.average(temp[6:11, 6:11]))           #bottom left
    poss.append(np.average(temp[6:11, 8:13]))           #bottom center
    poss.append(np.average(temp[6:11, 10:15]))          #bottom right
    poss.append(np.average(temp[8:13, 6:11]))           #center left
    poss.append(np.average(temp[8:13, 10:15]))          #center right
    poss.append(np.average(temp[10:15, 6:11]))          #top left
    poss.append(np.average(temp[10:15,8:13]))           #top center
    poss.append(np.average(temp[10:15,10:15]))          #top right  
    filtered = [i for i, x in enumerate(poss) if x < 0.5]
    if len(filtered) > 0:
        i_min = np.random.choice(np.array(filtered))
    else:
        i_min = np.argmin(np.array(poss))
    
    if i_min == 0:      end_pos = (-mag, -mag)
    elif i_min == 1:    end_pos = (0, -mag)
    elif i_min == 2:    end_pos = (mag, -mag)
    elif i_min == 3:    end_pos = (-mag, 0)
    elif i_min == 4:    end_pos = (mag, 0)
    elif i_min == 5:    end_pos = (-mag, mag)
    elif i_min == 6:    end_pos = (0, mag)
    elif i_min == 7:    end_pos = (mag, mag)
    return end_pos

def sense_end_of_path():
    xObs, yObs = get_sensors(0)
    end_of_path = False
    rMin = 2
    for i in range(len(xObs)):
        x = xObs[i]
        y = yObs[i]
        r = np.sqrt(x**2 + y**2)
        if r<2:
            end_of_path = True
            if r < rMin:
                xWorst = x
                yWorst = y
    if end_of_path:
        tWorst = np.arctan2(yWorst,xWorst)
        tBest = tWorst + np.pi
        X = 1*np.sin(tBest)
        Y = -1*np.cos(tBest)
        _, _, _ = move(X, Y, 0)  
    time.sleep(0.5)
    return end_of_path

def plot_choice(maze, best_itemNo = None):
        
    if best_itemNo is not None:

        #plot results
        xMin = 0
        xMax = len(maze[0])
        yMin = 0
        yMax = len(maze)
        xOffset = 10
        yOffset = 10
    
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        x = xMin - xOffset
        y = yMin - yOffset
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                val = maze[i][j]
                if val == 1:    ax1.scatter(x, y, color = 'black', marker='s', s = 10, zorder = 3)
                if val == 2:    ax1.scatter(x, y, color = 'red', marker='x', s = 10, zorder = 3)
                if val == 3:    ax1.scatter(x, y, color = 'blue', marker='s', s = 10, zorder = 3)
                if val == 7:    ax1.scatter(x, y, color = 'brown', marker='s', s = 10, zorder = 3)
                if val == 9:    
                    ax1.scatter(x, y, color = 'gold', marker='*', s = 50, zorder = 3)
                if val > 9:    
                        ax1.scatter(x, y, color = 'green', marker='*', s = 50, zorder = 3)
                        ax1.text(x,y, str(val - 10))
                x+=1
            y +=1
            x = xMin - xOffset
        xTicks = np.array(range(xMin-xOffset, xMax-xOffset))
        yTicks = np.array(range(yMin-yOffset, yMax-yOffset))
        ax1.grid(which='major', alpha=0.2, color ='black')
        
        ax1.set_xticks(xTicks)
        ax1.set_yticks(yTicks)
        ax1.axis([xMin-xOffset-0.5, xMax-xOffset-0.5, yMin-yOffset-0.5, yMax-yOffset-0.5])
        ax1.scatter(0,0, c ="grey", linewidths = 2, marker ="v", s = 200)
        ax1.scatter(0,0, c ="black", linewidths = 2, marker ="+", s = 100)
        ax1.set_aspect('equal', 'box')
   
        crops = os.listdir('./output/temp_count/images/1_cropped/')
        best_image = './output/temp_count/images/1_cropped/' + crops[best_itemNo]
        best_image = img.imread(best_image)
        ax2.imshow(best_image[::3,::3],cmap=cm.Greys)
        
        raw_path = './output/temp_count/images/0_raw/'
        raw_image = crops[best_itemNo].split('_')[0] + '_rgb.png'
        raw_image = raw_path + raw_image
        raw_image = img.imread(raw_image)
        ax3.imshow(raw_image[::3,::3],cmap=cm.Greys)   
        
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)        
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)        
        fig.show()
            
### ---------------------------------------------------------------------------
# MOVEMENT
def go(maze, best_item, end_pos):

    #initialize errors
    exa = 0.0
    eya = 0.0
    eta = 0.0
    iDist = 0
    
    #get path with astar
    path, maze, end_pos  = get_path(maze, end_pos) 
    logger.info('=> Pepper is moving (0,0) --> ({}, {})'.format(int(np.round(end_pos[0])), int(np.round(end_pos[1],0))))


    if len(path) == 0: 
        path = [(10, 10), (9, 10)]  #back up one
    
    #get actions from path
    actions = get_actions(path, end_pos, best_item)
    
    #follow actions
    for ia in range(len(actions)):
        ux  = float(actions[ia][0])
        uy  = float(actions[ia][1])
        ut  = float(actions[ia][2])
        exa, eya, eta = controller(ux, uy, ut)  
        if exa == None: break 
        else:
            iDist += np.sqrt(ux**2 + uy**2)
    if exa == None: 
        write_dist(iDist)
        return
    else:
        if best_item is not None: declare(best_item)
        write_dist(iDist)
        return

def controller(ux, uy, ut):  
    if not sense_end_of_path():
        exa1, eya1, eta1 = 0.0, 0.0, 0.0
        exa2, eya2, eta2 = 0.0, 0.0, 0.0
        
        if np.abs(ut)>0.1:                              #rotate
            if np.abs(ut)<0.3: ut = ut/np.abs(ut)*0.3   #min for "complete" move
            exa1, eya1, eta1 = move(0, 0, ut) 
            if exa1 == None:  
                return None, None, None
        if np.abs(ux)>0.1 or np.abs(uy) > 0.1:          #translate   
            exa2, eya2, eta2 = move(ux, uy, 0)  
            if exa2 == None: 
                return None, None, None
            return exa2, eya2, eta1
        else: 
            return 0.0, 0.0, eta1
    else:   
        return None, None, None
    
def move(ux, uy=0, ut=0):
    # end pos absolute
    xF, yF, tF = end_pos(ux, uy, ut)
    
    # do movement
    ux_m, uy_m, ut_rad = blocks_to_SI(ux, uy, ut)
    completed_move = _mot.moveTo(ux_m, uy_m, ut_rad)    # do movement
    
    if completed_move: 
        return 0.0, 0.0, 0.0
    else:
        #slowing down to complete move      
        counter = 0        
        while not completed_move and counter < 20:
            #get relative error
            ex, ey, et = rel_error(xF, yF, tF)
            ex_m, ey_m, et_rad = blocks_to_SI(ex, ey, et)
            
            #do movement
            completed_move = _mot.moveTo(-ex_m, -ey_m, 0.0, _mot.getMoveConfig("Min"))
            _mot.waitUntilMoveIsFinished()
            time.sleep(0.5)
            counter +=1
        
        if completed_move:
            return 0.0, 0.0, 0.0
        else:
            speak('Please fix my base so I can continue.')
            raw_input('Fix Pepper base. Press Enter to continue.')
            return None, None, None    

def move_fail(val):
    try:
        obsX = np.round(val[2][0],3)
        obsY = np.round(val[2][1],3)
        temp = _mot.getRobotPosition(True)
        rX = np.round(temp[0],3)
        rY = np.round(temp[1],3) 
        x_rel = obsX - rX
        y_rel = obsY - rY
        val = np.round(np.sqrt(x_rel**2 + y_rel**2),4)*1000.
        logger.info('=> Pepper cannot move due to object within {} mm.'.format(val)) 
    except: 
        logger.info('=> Pepper cannot move: {}'.format(val)) 
        
# ----------------------------------------------------------------------------
# MAIN
def run():
    iRT = 0
    iTT = 0
    best_item = None
    while best_item is None:    
        
        speak('I am going to explore')
        
        #quick look around (sensor only)
        train_t0 = np.round(time.time(), 3)    
        maze, best_item, end = quick_check()
        train_t1 = np.round(time.time(), 3) 
        iTT += (train_t1 - train_t0)
        
        #go to empty space
        run_t0 = np.round(time.time(), 3)    
        go(maze, best_item, end)   
        run_t1 = np.round(time.time(), 3)    
        iRT += (run_t1 - run_t0)
        
        #full look around (sensor and vision)
        train_t0 = np.round(time.time(), 3)    
        maze, best_item, end = check()   
        train_t1 = np.round(time.time(), 3)    
        iTT += (train_t1 - train_t0)
        
        
        if end is not None: 
            speak('Lets get an item over here!')
            
            run_t0 = np.round(time.time(), 3)    
            go(maze, best_item, end )
            run_t1 = np.round(time.time(), 3)    
            iRT += (run_t1 - run_t0)
            
    write_runTime(iRT)
    write_trainTime(iTT)
    
    
def ask(iInc):
    
    #rules
    # 0) if the image only contains ONE item: train that item, continue
    # 1) if the item requested is ANYWHERE in the cropped image: train = asked, continue
    # 2) if there are more than three items: continue
    # 3) if image has 2-3 items, judge whether ONE item contains over 50% of image: 
    #           a) if yes: train that item
    #           b) if no: assign -1 and keep looking
    
    asked_item = raw_input('Requested class number: ')    
    train_item = raw_input('Actual class number: ')
    
    #overwrite the current train item every time
    filepath = './output/temp_count/current_train_item.txt'
    f = open('./output/temp_count/current_train_item.txt', "w")
    f.writelines(str(train_item))
    f.close()
    
    #make the guess_true record, then append to it
    filepath = './output/temp_count/guess_true_record.txt'
    if not os.path.isfile(filepath):
        f = open(filepath, "w")
        f.writelines('')
        f.close()
    
    #correct prediction?
    correct = (asked_item == train_item)
    if correct:     score = 1
    else:           score = 0
    
    f = open(filepath, "a")
    temp = '{}, {}, {}, {}\n'.format(iInc, asked_item, train_item, score)
    f.write(temp)
    
    return train_item
    
def train():
    train_t0 = np.round(time.time(), 3)    
    subprocess.Popen("conda run -n py38 python models/cbcl/cbcl_update.py".split(), stdout=subprocess.PIPE)
    train_t1 = np.round(time.time(), 3) 
    iTT = train_t1 - train_t0
    write_trainTime(iTT)
    time.sleep(3)

    filepath = './output/temp_count/min_remaining.txt'
    if os.path.isfile(filepath):
        f = open(filepath, "r")
        txt = str(f.read())
        f.close()
        logger.info(txt)
    
def test(inc, total_inc):
    speak('I am going to test my knowledge.')
    logger.info('=> Pepper is testing knowledge of environment.')
    subprocess.Popen("conda run -n py38 python models/cbcl/cbcl_test.py".split(), stdout=subprocess.PIPE)
    time.sleep(3)
    filepath = './output/temp_count/current_acc.txt'
    if os.path.isfile(filepath):
        f = open(filepath, "r")
        acc = float(f.read())
        f.close()
        logger.info('=> Pepper scored a {} percent accuracy.'.format(np.round(acc*100.,1)))
    speak('I have finished iteration {}'.format(inc))
    logger.info('=> Pepper has finished iteration {} of {}'.format(inc, total_inc))
    batLev = _bat.getBatteryCharge()
    logger.info("=> Pepper is {}% charged.".format(batLev))

def cleanup():
 
    dir = './output/temp_cbcl/'
    for f in os.listdir(dir): 
        item = os.path.join(dir, f)
        os.remove(item)
    
    dir = './output/temp_count/'
    for f in os.listdir(dir): 
        item = os.path.join(dir, f)
        if item.endswith('.txt'): os.remove(item)
    
    dir = './output/temp_count/images/0_raw/'
    for f in os.listdir(dir): 
        item = os.path.join(dir, f)
        os.remove(item)

    dir = './output/temp_count/images/1_cropped/'
    for f in os.listdir(dir): 
        item = os.path.join(dir, f)
        os.remove(item)
    
    dir = './output/temp_count/images/2_guesses/'
    for f in os.listdir(dir): 
        item = os.path.join(dir, f)
        os.remove(item)

#events
noMove = _mem.subscriber('ALMotion/move_failed')
noMove.signal.connect(move_fail)