# Active Class Selection for Few-Shot Class-Incremental Learning
Code for the paper: ***in review***

## Abstract
Previous work has shown that data acquisition is made more efficient with active class selection (ACS), where a learner may actively request more training data about a specific class. Additionally, processing and memory efficiency are important considerations for few-shot class-incremental learning (FSCIL), where a learner receives limited training data in increments, rather than all at once. Motivated to create robots that learn to recognize unknown classes of objects in unknown environments, we propose a solution that combines the efficiency of data acquisition in ACS with the computational and memory efficiency of FSCIL. As an extension of Centroid-Based Concept Learning with Pseudo-Rehearsal (CBCL-PR), our algorithm uses cluster statistics to actively select new classes of data to learn. We show that not only does CBCL-PR show state-of-the-art performance in a pure few-shot class-incremental learning setting, but also that cluster-based active class selection improves performance over previous methods in two environments: Minecraft simulation and with a real robot.

## Experiments
We ran three experiments to test active class selection on agent/robot learning:
+ [Agent in Minecraft simulation](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/minecraft)
+ [Robot in an indoor environment](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/pepper)
+ [Agent with no environment (batch setting)](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/batch)

## Data 
We used three standard datasets in our tests:
+ [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html).
+ [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset).
+ [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/).

## Other reference
+ This work builds from CBCL-PR, an extension of [CBCL](https://github.com/aliayub7/CBCL).
+ This work also incoporates YOLO object detection 

**Usage.**  To run this code, you should use the following steps:
1. Download the Malmo environment. 
2. Download the image data and put it into ./utils/data
3. Run the ./utils/get_features.py to extract features to ./utils/features
4. Open a Minecraft client and run quick-test.py. *Her
e, you can visualize the potential field as the agent goes through the environement.*
5. Open many Minecraft clients and run main.py (with intended parameters) to run the full test. *Here, the results are written to ./results/[date]/*






