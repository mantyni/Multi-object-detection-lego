# run this before:
# python3 detect.py --source lego_dataset/images/test/ --weights runs/train/yolo_lego_det10/weights/best.pt --conf 0.25 --name yolo_lego_det


import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


detections_dir = "runs/detect/yolo_lego_V2_det3/"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]

random_detection_image = Image.open(random.choice(detection_images))

plt.imshow(np.array(random_detection_image))
