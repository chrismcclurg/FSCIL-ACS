# ACS-FSCIL in Minecraft
In an iteration, a Minecraft agent has two minutes to search the environment for new visual examples of objects. The agent navigates with an internal potential field, created from objects within an observable distance. The agent can observe visual examples of an object *only* when it stands over that object. After the interval of searching, the agent processes the visual examples by updating its cluster space (CBCL-PR) or re-training on all of the previous training data (SVM). Finally, the agent makes predictions on the test data (static subset of original dataset) and classification accuracy is recorded. The agent's affinity to different classes of items is updated using the ACS methods, which directly affect the future potential field for navigation. The experiment continues for 360 minutes. 

<img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/minecraft_flowchart.png"> 

## Data 
We used one standard dataset in this test. Download and put in the `./pepper/utils/data/` folder.
+ [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset)

## Other reference
+ We build from CBCL-PR, an extension of [CBCL](https://github.com/aliayub7/CBCL).
+ We use [YOLO v3](https://github.com/arunponnusamy/object-detection-opencv) for object detection.
+ We use pre-trained [weights](https://pjreddie.com/media/files/yolov3.weights) for the YOLO model. Download and put in the `./pepper/models/yolo/` folder.

## If you consider citing us
```
This paper is currently in review. 
```
