# ACS-FSCIL in Minecraft
In an iteration, a Minecraft agent has two minutes to search the environment for new visual examples of objects. The agent navigates with an internal potential field, created from objects within an observable distance. The agent can observe visual examples of an object *only* when it stands over that object. After the interval of searching, the agent processes the visual examples by updating its cluster space (CBCL-PR) or re-training on all of the previous training data (SVM). Finally, the agent makes predictions on the test data (static subset of original dataset) and classification accuracy is recorded. The agent's affinity to different classes of items is updated using the ACS methods, which directly affect the future potential field for navigation. The experiment continues for 360 minutes. 

<img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/minecraft_flowchart.png"> 

## Preparation
1. Download the [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset) and [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset)dataset and put in the `./pepper/utils/data/` folder.
2. Run **./pepper/utils/get_features.py** to extract features into the `./pepper/utils/features/` folder.
3. Change parameters (classifier, active class selection type, etc.) in **./pepper/utils/params.txt** for your test.
4. Download the pre-trained [weights](https://pjreddie.com/media/files/yolov3.weights) for YOLO and put in the `./pepper/models/yolo/` folder.

## Notes
+ This test could easily be extended to other datsets. The four steps above would be the same; however, data-specific files need to be added to the `./pepper/utils/` folder (as currently done) and appropriate referencing in the code. 
+ The IP address in **./pepper/models/pepper.py** needs modified to be your local network, for which Pepper is also connected.
+ The main script requires [Python 2.7-32 bit](https://www.python.org/downloads/) (for Windows users). 
+ To deal with the memory limit (2GB) of Windows 32-bit applications, **./pepper/models/pepper.py** spawns new subprocesses for YOLO and CBCL models in a virtual environment with Python 3.8, which is designated py38.

## Reference
+ We build from CBCL-PR, an extension of [CBCL](https://github.com/aliayub7/CBCL).
+ We use [YOLO v3](https://github.com/arunponnusamy/object-detection-opencv) for object detection.

## If you consider citing us
```
This paper is currently in review. 
```
