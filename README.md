# FSCIL with Active Class Selection (ACS) in Minecraft

**Overview.** This code is for implementing CBCL with active class selection in the Minecraft (Malmo) environment. 

**Scenerio.** The robot passes through the environment with partial observability. When a Minecraft item is within range, an item is identified. A potential field is used to enable the robot to navigate from one desirable item to another. When the agent gets within close proximity to an item, the agent can 'collect' exemplars of this item. In reality, these exemplars are mapped to a real-world image dataset (CIFAR-100, GROCERY STORE, CUB-200), so the agent is collecting feature vectors. The robot updates its understanding of the world (cluster space). The cluster space is used to inform the most desirable classes.

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






