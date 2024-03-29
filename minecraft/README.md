# ACS-FSCIL in Minecraft
<p align="justify">
In an iteration, a Minecraft agent has two minutes to search the environment for new visual examples of objects. The agent navigates with an internal potential field, created from objects within an observable distance. The agent can observe visual examples of an object *only* when it stands over that object. After the interval of searching, the agent processes the visual examples by updating its cluster space (CBCL-PR) or re-training on all of the previous training data (SVM). Finally, the agent makes predictions on the test data (static subset of original dataset) and classification accuracy is recorded. The agent's affinity to different classes of items is updated using the ACS methods, which directly affect the future potential field for navigation. The experiment continues for 360 minutes. 
</p>


<p align="center">
  <img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/minecraft_flowchart.png" width=80% height=80%> 
</p>
  

## Preparation
1. Download the [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets and put in the `./minecraft/utils/data/` folder.
2. Run **./minecraft/utils/get_features.py** to extract features into the `./minecraft/utils/features/` folder.
3. Download [Project Malmo](https://github.com/microsoft/malmo). See note below.

## Notes
+ Make sure that the version of Project Malmo corresponds to the version of Python. We found Python 3.6 to be the easiest to use.   
+ This test could easily be extended to other datsets. The steps above would be the same; however, data-specific files need to be added to the `./minecraft/utils/` folder (as currently done). Specific adjustments include: 
  + Tabulating the fine and coarse labels in **./minecraft/utils/env/[data]-labels.xlsx**
  + Running a script **./minecraft/utils/get_env-[data].py** to get item placement as **./minecraft/utils/env/[data]-mapping.xlsx**
  + Searching for any data-specific references in the current code. There should not be many.
+ There are two ways to run the simulation. 
  + The **./minecraft/quick-test.py** runs a single process, plotting in real time the potential field for navigation. 
  + The **./minecraft/main.py** runs the full test of specified test conditions, for which you can specify the number of processors you would like to use.
  
## Results
Our results for FSCIL-ACS in Minecraft are shown below. Active class selection and classifier (CBCL-PR or SVM) are varied.
<p align="center">
  <img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/minecraft_results.png" width=80% height=80%>
</p>

## Reference
+ We build from CBCL-PR, an extension of [CBCL](https://github.com/aliayub7/CBCL).
+ We use the [Project Malmo](https://github.com/microsoft/malmo) platform for Minecraft testing. Additional resources can be found on Discord.

## If you consider citing us
```
@inproceedings{mcclurg2023acsfscil,
  title={Active Class Selection for Few-Shot Class-Incremental Learning},
  author={McClurg, Christopher and Ayub, Ali and Tyagi, Harsh and Rajtmajer, Sarah M. and Wagner, Alan R.},
  booktitle={Conference on Lifelong Learning Agents},
  year={2023}}
```
