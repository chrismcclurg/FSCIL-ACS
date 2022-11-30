# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================

import numpy as np
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return "{} - g: {} h: {} f: {}".format(self.position,self.g,self.h, self.f)

    def __lt__(self, other):
      return self.f < other.f
    
    def __gt__(self, other):
      return self.f > other.f    
    
def return_path(current_node, maze, start, end, xOffset, yOffset):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    maze[start] = 2
    maze[end] = 2
    for p in path[1:-1]:  maze[p] = 3
    ans = [path[::-1], maze]
    xe = end[1] - xOffset
    ye = end[0] - yOffset
    ans = [path[::-1], maze, (xe, ye)]
    return ans
    
def get_path(maze, end_pos):
    
    #make maze binary
    maze[maze > 1] = 1
    
    #map coordinates
    xOffset = 10
    yOffset = 10
    
    #start node
    start = (0 + yOffset, 0 + xOffset)
            
    #end node position
    xe = int(np.round(end_pos[0],0))
    ye = int(np.round(end_pos[1],0))
    end = (ye + yOffset, xe + xOffset)
    
    #correct end node if necessary
    end = nearest_pos(end, maze)
          
    #create start, end nodes
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    
    current_node = None

    #initialize open, closed lists
    open_list = []
    closed_list = []

    #heapify the open_list and add start node
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)

    #add a stop condition
    outer_iterations = 0
    max_iterations = (len(maze[0]) * len(maze) // 2)

    #neigboring squares searched
    # adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0))
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)
    #find path
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
          print("=> Astar pathfinding failed to complete (too many iterations).")
          return return_path(current_node, maze, start, end, xOffset, yOffset)       
        
        #get current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        #found the goal
        if current_node == end_node:
            return return_path(current_node, maze, start, end, xOffset, yOffset)

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    print("=> Astar pathfinding failed to complete (couldn't get path).")
    xe = end[1] - xOffset
    ye = end[0] - yOffset
    return [], maze, (xe, ye)

def nearest_pos(pos, maze):
    node = pos
    nx0 = pos[1]
    nz0 = pos[0]
    
    # print(nx0 - 10, nz0 - 10)

    
    #node limits
    xmax = len(maze[0]) -1
    xmin = 0
    zmax = len(maze) -1
    zmin = 0
    
    #correct node if not in map
    if nx0 > xmax:    nx0 = xmax
    elif nx0 < xmin:  nx0 = xmin
    
    if nz0 > zmax:    nz0 = zmax
    elif nz0 < zmin:  nz0 = zmin
    
    # print(nx0 - 10, nz0 - 10)

       
    #search for OK node
    if maze[(nz0, nx0)] == 1:
        search = True
        k = 1
        while search:
            for j in range(0,k+1):
                for i in range(-k, k+1):
                    
                    nz_temp = nz0 + j
                    nx_temp = nx0 + i
                    
                    if nx_temp > xmax:    nx_temp = xmax
                    elif nx_temp < xmin:   nx_temp = xmin
                    
                    if nz_temp > zmax:    nz_temp = zmax
                    elif nz_temp < zmin:   nz_temp = zmin
                    
                    node_temp = (nz_temp, nx_temp)
                    # print(nx_temp - 10, nz_temp - 10)
                    
                    if maze[node_temp] == 0:
                        node = node_temp                
                        search = False      
                    if search is False: break
                if search is False: break
            for j in range(0, -(k+1), -1):
                for i in range(-k, k+1):
                    
                    nz_temp = nz0 + j
                    nx_temp = nx0 + i
                    
                    if nx_temp > xmax:    nx_temp = xmax
                    elif nx_temp < xmin:   nx_temp = xmin
                    
                    if nz_temp > zmax:    nz_temp = zmax
                    elif nz_temp < zmin:   nz_temp = zmin
                    
                    node_temp = (nz_temp, nx_temp)                    
                    if maze[node_temp] == 0:
                        node = node_temp                
                        search = False      
                    if search is False: break
                if search is False: break
            k+=1
    return node

def get_actions(path, end_pos, best_item):
    ans     = []   
    rCurr   = 0
    rSum    = 0
    tPrev   = 90. #looking straight ahead
    tCurr   = 90. #looking straight ahead
    
    xF      = end_pos[0]
    yF      = end_pos[1]
    tF      = np.round(np.arctan2(yF, xF) *180. / np.pi,0)
    
    x1 = 0
    y1 = 0
    distance_to_target = np.sqrt((yF - y1)**2 + (xF - x1)**2 )
    # rotation_to_target = tF - tCurr
    
    for ip in range(1,len(path)): 
        
        #count nodes
        dx = path[ip][1] - path[ip-1][1]
        dy = path[ip][0] - path[ip-1][0]
        x1 += dx
        y1 += dy
        distance_to_target = np.sqrt((yF - y1)**2 + (xF - x1)**2)
        if distance_to_target < 3 and best_item is not None: break
        tCurr = np.round(np.degrees(np.arctan2(dy, dx)),2)
        rCurr = np.round(np.sqrt(dx**2 + dy**2),2)
        
        #change in direction
        if tCurr != tPrev: 
            if rSum > 0: 
                ans.append([np.round(rSum,2), 0, 0])     
                rSum = 0
            ans.append([0, 0, tCurr - tPrev])
            tPrev = tCurr
        rSum += rCurr
        
        #distance limitter
        if rSum >= 3.0: 
            ans.append([np.round(rSum,2), 0, 0])     
            rSum = 0     
            
    #final translation
    if rSum > 0.0: 
        ans.append([np.round(rSum,2), 0, 0])  
    
    #final rotation
    if tCurr != tF: 
        ans.append([0, 0, tF - tCurr])          
        
    return ans

def plot_map(maze):
    #map coordinates
    xMin = 0
    xMax = len(maze[0])
    yMin = 0
    yMax = len(maze)
    
    #plot map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xTicks = range(xMin, xMax+1)
    yTicks = range(yMin, yMax+1)
    ax.set_xticks(xTicks)
    ax.set_yticks(yTicks)
    ax.grid(which='major', alpha=0.2, color ='black')
    ax.axis([xMin, xMax, yMin, yMax])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    #plot nodes
    x = xMin + 0.5
    y = yMin + 0.5
    
    for row in maze: 
        for val in row:
            if val == 1:    plt.scatter(x, y, color = 'black', marker='s', s = 25, zorder = 3)
            elif val == 2:  
                plt.scatter(x, y, color = 'red', marker='x', s = 25, zorder = 5)
            elif val == 3:  
                plt.scatter(x, y, color = 'blue', marker='s', s = 25, zorder = 4)  
            elif val == 8:  
                plt.scatter(x, y, color = 'pink', marker='s', s = 25, zorder = 4)                  
            x+=1
        y +=1
        x = xMin + 0.5
    


