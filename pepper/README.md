# ACS-FSCIL with Pepper
Pepper is given sixty iterations to search the environment for new visual examples of objects. An iteration consists of Pepper (1) relocating, (2) searching, (3) choosing an object, and (4) receiving training examples. To relocate, Pepper first rotates with range sensors to define a localized map; an end location is chosen among the free space, and A* path planning is used. To search, Pepper uses a top camera, which provides up to 2560x1080 pixel resolution at 5 fps. After taking images of the surrounding area, Pepper uses the [YOLO](https://pjreddie.com/darknet/yolo/) algorithm pre-trained on the [Microsoft COCO](https://cocodataset.org/#home). To choose an object, Pepper uses [CBCL-PR](https://github.com/aliayub7/CBCL) for (initially weaker) classification with active class selection to pick the most desirable class. To receive training examples, Pepper shows the human helper (experimenter) an image of the desired class, for which the human helper can give the true label of the predicted class, as well as ten visual examples. After every iteration, Pepper updates her cluster space of learned classes. The robot's affinity to different classes of items is updated using the ACS methods. At the end of every three iterations, Pepper makes predictions on the test data and classification accuracy is recorded.

<p align="center">
  <img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/pepper_flowchart.png"> 
</p>

## Preparation
1. Download the [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset) dataset and put in the `./pepper/utils/data/` folder.
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
