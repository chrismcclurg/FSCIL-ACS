# Active Class Selection for FSCIL
Previous work has shown that data acquisition is made more efficient with active class selection (ACS), where a learner may actively request more training data about a specific class. Additionally, processing and memory efficiency are important considerations for few-shot class-incremental learning (FSCIL), where a learner receives limited training data in increments, rather than all at once. Motivated to create robots that learn to recognize unknown classes of objects in unknown environments, we propose a solution that combines the efficiency of data acquisition in ACS with the computational and memory efficiency of FSCIL. As an extension of Centroid-Based Concept Learning with Pseudo-Rehearsal (CBCL-PR), our algorithm uses cluster statistics to actively select new classes of data to learn. We show that not only does CBCL-PR show state-of-the-art performance in a pure few-shot class-incremental learning setting, but also that cluster-based active class selection improves performance over previous methods in two environments: Minecraft simulation and with a real robot.

<img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/malmo_iso.jpg" width=45% height=45%> <img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/pepper_iso.jpg" width=45% height=45%>


## Experiments
We ran three experiments to test active class selection on agent/robot learning:
+ [Agent in Minecraft simulation](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/minecraft)
+ [Robot in an indoor environment](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/pepper)
+ [Agent with no environment (batch setting)](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/batch)

## Data 
We used three standard datasets in our tests:
+ [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
+ [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset)
+ [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)

## Other reference
+ This work builds from CBCL-PR, an extension of [CBCL](https://github.com/aliayub7/CBCL).
+ This work uses YOLO v3 for object detection. The corresponding repo can be found [here](https://github.com/arunponnusamy/object-detection-opencv).

## If you consider citing us
```
This paper is currently in review. 
```






