# Active Class Selection for Few-Shot Class-Incremental Learning
Code for the paper: _[in review].

## Abstract
Previous work has shown that data acquisition is made more efficient with active class selection (ACS), where a learner may actively request more training data about a specific class. Additionally, processing and memory efficiency are important considerations for few-shot class-incremental learning (FSCIL), where a learner receives limited training data in increments, rather than all at once. Motivated to create robots that learn to recognize unknown classes of objects in unknown environments, we propose a solution that combines the efficiency of data acquisition in ACS with the computational and memory efficiency of FSCIL. As an extension of Centroid-Based Concept Learning with Pseudo-Rehearsal (CBCL-PR), our algorithm uses cluster statistics to actively select new classes of data to learn. We show that not only does CBCL-PR show state-of-the-art performance in a pure few-shot class-incremental learning setting, but also that cluster-based active class selection improves performance over previous methods in two environments: Minecraft simulation and with a real robot.

## Experiments
+ For a the application in Minecraft, click ![here]

**Scenerio.** The robot passes through the Minecraft environment with partial observability. When a Minecraft item is within range, the item is identified. A potential field, created from the sum of unique observations, is used to navigate the robot from one desirable item to another. When the agent gets within close proximity to an item, the agent can 'collect' exemplars of this item. In reality, these exemplars are mapped to a real-world image dataset (CIFAR-100, GROCERY STORE, CUB-200), so the agent is collecting feature vectors of a certain class. The robot updates its understanding of the world (cluster space). The cluster space is used to inform the most desirable classes.

![An overview of the model](https://github.com/chrismcclurg/minecraft-incremental-learning/blob/main/compModel.png?raw=true)

**Usage.**  To run this code, you should use the following steps:
1. Download the Malmo environment. 
2. Download the image data and put it into ./utils/data
3. Run the ./utils/get_features.py to extract features to ./utils/features
4. Open a Minecraft client and run quick-test.py. *Here, you can visualize the potential field as the agent goes through the environement.*
5. Open many Minecraft clients and run main.py (with intended parameters) to run the full test. *Here, the results are written to ./results/[date]/*

**Project Malmo.** To download and/or resolve any issues with the Malmo environment, please refer to https://github.com/microsoft/malmo.

**Datasets.** The datasets can be found in the following locations:
- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- GROCERY STORE: https://github.com/marcusklasson/GroceryStoreDataset






