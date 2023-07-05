# Active Class Selection for FSCIL
<p align="justify">For real-world applications, robots will need to continually learn in their environments through limited interactions with their users. Toward this, previous works in few-shot class incremental learning (FSCIL) and active class selection (ACS) have achieved promising results but were tested in constrained setups. Therefore, in this paper, we combine ideas from FSCIL and ACS to develop a novel framework that can allow an autonomous agent to continually learn new objects by asking its users to label only a few of the most informative objects in the environment. To this end, we build on a state-of-the-art (SOTA) FSCIL model and extend it with techniques from ACS literature. We term this model Few-shot Incremental Active class SeleCtiOn (FIASco). We further integrate a potential field-based navigation technique with our model to develop a complete framework that can allow an agent to process and reason on its sensory data through the FIASco model, navigate towards the most informative object in the environment, gather data about the object through its sensors and incrementally update the FIASco model. Experimental results on a simulated agent and a real robot show the significance of our approach for long-term real-world robotics applications.</p>

<p align="center">
  <img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/malmo_iso.jpg" width=48% height=48%> <img src="https://github.com/chrismcclurg/FSCIL-ACS/blob/main/img/pepper_iso.jpg" width=48% height=48%>
</p>

## Experiments
We ran two experiments to test active class selection on agent/robot learning:
+ [Agent in Minecraft simulation](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/minecraft)
+ [Robot in an indoor environment](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/pepper)
<!--- + [Agent with no environment (batch setting)](https://github.com/chrismcclurg/FSCIL-ACS/tree/main/batch) --->

## Data 
We used three standard datasets in our tests. Download each one and put in the desired `./*/utils/data/` folder.
+ [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
+ [GROCERY STORE](https://github.com/marcusklasson/GroceryStoreDataset)
+ [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)


## Other
+ We build from CBCL-PR, an extension of [CBCL](https://github.com/aliayub7/CBCL), a classifier for FSCIL.
+ We use [YOLO v3](https://github.com/arunponnusamy/object-detection-opencv) for object detection.
+ We use the [Project Malmo](https://github.com/microsoft/malmo) platform for Minecraft testing.
+ We use pre-trained [weights](https://pjreddie.com/media/files/yolov3.weights) for the YOLO model. Download and put in the `./pepper/models/yolo/` folder.

## If you consider citing us
```
@inproceedings{mcclurg2023acsfscil,
  title={Active Class Selection for Few-Shot Class-Incremental Learning},
  author={McClurg, Christopher and Ayub, Ali and Tyagi, Harsh and Rajtmajer, Sarah M. and Wagner, Alan R.},
  booktitle={Conference on Lifelong Learning Agents},
  year={2023}}
```






