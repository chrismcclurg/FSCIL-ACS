# FSCIL with Active Class Selection (ACS) in Minecraft

**Overview** This code is for implementing CBCL with active class selection in the Minecraft (Malmo) environment. 

**Scenerio.** The robot passes through the environment with partial observability. When a Minecraft item is within range, an item is identified. A potential field is used to enable the robot to navigate from one desirable item to another. When the agent gets within close proximity to an item, the agent can 'collect' exemplars of this item. In reality, these exemplars are mapped to a real-world image dataset (CIFAR-100, GROCERY STORE, CUB-200), so the agent is collecting feature vectors. The robot updates its understanding of the world (cluster space). The cluster space is used to inform the most desirable classes.

**Dependencies.**  The following 



