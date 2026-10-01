# import pybullet
# import pybullet_data
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import random 
import math
import csv
from scipy.spatial.transform import Rotation as R
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn 
from torchvision import models
from torchsummary import summary
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torchvision import transforms as T
# import pybullet_utils.bullet_client as bc
import argparse
import os.path as osp

img_path = 'Figure_1.png'

resnet50  = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

class Net(nn.Module):
    def __init__(self, resnet50):
        super(Net, self).__init__()
        self.pretrain = resnet50
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    
    def forward(self, x):
        x = self.pretrain(x)
        print(x.shape)
        x= x.view(x.size(0),-1)
        print(x.size(0))
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

model = Net(resnet50)
model.to('cuda:1')
model.eval()

img = Image.open(img_path).convert('RGB')
preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )])
img = preprocess(img)
Img = torch.unsqueeze(img, 0).to('cuda:1')
print(Img.shape)

pred = model(Img)
print(pred)