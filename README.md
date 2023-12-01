# Lego multi object detection
Synthetic lego dataset for testing multi object detection using machine learning. Annotations saved in PASCAL-VOC format. 
Blender scripts with detailed annotations are provided.

#### YoloV5 detection on synthetic dataset

![Alt text](images/00707_1.jpg "00707_1")
![Alt text](images/00707.jpg "00707")

![Alt text](images/00724_1.jpg "00724_1")
![Alt text](images/00724.jpg "00724")


#### YoloV5 detection on real world lego parts (trained on synthetic dataset)
![Alt text](images/webcam1.gif "Lego")

#### 

## Instructions

### Important - Blender version compatibility: 
Currently Lego rendering code is tested with Blender 2.93 & 3.65 versions.
Newer Blender versions are not backward compatible, which means version 2.X won't work with 3.X and same for 4.X versions. Working Blender file for specific version will be in [blender directory](blender/)
 in this repo. I will try to update the repository to maintain compatibility with latest Blender versions, but if you have any problems running the code please create a github issue and will respond as soon as I can.

### 1. Creating synthetic Lego part dataset using Blender 

![Alt text](images/blender.jpg "Blender")

1. Install required packages for internal Blender Python library to run Blender Python scripts (change to your Blender version below):
	1. cd to your Blender Python dir `cd ~/blender-2.93.4-linux-x64/2.93/python`
	2. install ensurepip `./bin/python3.9 -m ensurepip`
	3. install pip `./bin/python3.9 -m pip install -U pip`
	4. install pascal-voc-writer for annotation extraction `./bin/python3.9 -m pip pascal-voc-writer`
2. Copy the contents from `blender` from this repo into to your Blender directory, for eg. ~/home/blender-2.93.4-linux-x64.
3. Based on your version use Blender to open `Blender_2.93_Lego_rendering_scene.blend` or `Blender_3.65_Lego_rendering_scene.blend` (in your Blender home directory).
4. In Blender click on the scripting tab and press run script button to create the Lego dataset. Run the script multiple times to create more Lego renderings in batches (each batch is limited to a few hundred images due to blender memory leaks).
5. Annotations are in PASCAL-VOC format and images in JPG 300x300 format, saved in `<your blender directory>/lego_renders`

### 2. Training Multi-object detection with YoloV5 

Using Jupyter notebook open yolov5.ipynb and execute the cells. 

### 3. Training Multi-object detection with SSD

Using Jupyter notebook open ssd_pytorch.ipynb and execute the cells. 


