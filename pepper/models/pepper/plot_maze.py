# =============================================================================
# Active Class Selection with Few-Shot Class Incremental Learning (FSCIL-ACS)
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer, N Tyagi
# =============================================================================

from env import get_path, get_actions, plot_map
import numpy as np 

maze = np.array([[1., 7., 7., 7., 7., 1., 7., 7., 7., 1., 1., 1., 0., 0., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 7., 1., 1., 1., 7., 1., 1., 0., 0., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 7., 1., 1., 1., 7., 7., 1., 7., 1., 0., 1., 1., 7.,
        7., 1., 1., 1., 1.],
       [1., 1., 1., 1., 0., 1., 7., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 7., 7.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 7., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 7., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 7., 7., 0., 0., 0., 0., 0., 7., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [7., 1., 7., 1., 1., 7., 7., 0., 0., 0., 0., 0., 7., 7., 1., 1.,
        1., 1., 1., 1., 1.],
       [0., 0., 7., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 7., 1., 1.,
        1., 1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 7., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 7., 1., 1.,
        1., 1., 1., 1., 1.],
       [7., 1., 1., 1., 1., 1., 1., 7., 7., 7., 7., 7., 7., 7., 7., 1.,
        1., 1., 1., 1., 1.],
       [7., 1., 1., 1., 7., 7., 1., 1., 7., 7., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 7., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.]])

maze[12][12] = 8

# end_pos = (-0.40022070350881944, 2.7005042841074895)
# path, maze, end_pos  = get_path(maze, end_pos)
# actions = get_actions(path, end_pos)

def get_freeSpace(maze, mag = 5):
    temp = maze.copy()
    temp[temp > 1] = 1    
    poss = []
    poss.append(np.average(temp[6:11, 6:11]))         #bottom left
    poss.append(np.average(temp[6:11, 8:13]))        #bottom center
    poss.append(np.average(temp[6:11, 10:15]))       #bottom right
    poss.append(np.average(temp[8:13, 6:11]))        #center left
    poss.append(np.average(temp[8:13, 10:15]))      #center right
    poss.append(np.average(temp[10:15, 6:11]))       #top left
    poss.append(np.average(temp[10:15,8:13]))       #top center
    poss.append(np.average(temp[10:15,10:15]))      #top right  
    print(poss)
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


end_pos = get_freeSpace(maze)
end_pos = (-6, 1)
path, maze, end_pos = get_path(maze, end_pos)
actions = get_actions(path, end_pos, None)
plot_map(maze)