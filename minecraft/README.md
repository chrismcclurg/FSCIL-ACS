# ACS-FSCIL in Minecraft
Pepper is given sixty iterations to search the environment for new visual examples of objects. An iteration consists of Pepper (1) relocating, (2) searching, (3) choosing an object, and (4) receiving training examples. To relocate, Pepper first rotates with range sensors to define a localized map; an end location is chosen among the free space, and A* path planning is used. To search, Pepper uses a top camera, which provides up to 2560x1080 pixel resolution at 5 fps. After taking images of the surrounding area, Pepper uses the [YOLO](https://pjreddie.com/darknet/yolo/) algorithm pre-trained on the [Microsoft COCO](https://cocodataset.org/#home). To choose an object, Pepper uses [CBCL-PR](https://github.com/aliayub7/CBCL) for (initially weaker) classification with active class selection to pick the most desirable class. To receive training examples, Pepper shows the human helper (experimenter) an image of the desired class, for which the human helper can give the true label of the predicted class, as well as ten visual examples. After every iteration, Pepper updates her cluster space of learned classes. The robot's affinity to different classes of items is updated using the ACS methods. At the end of every three iterations, Pepper makes predictions on the test data and classification accuracy is recorded.

<img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/minecraft_flow.png"> 

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
