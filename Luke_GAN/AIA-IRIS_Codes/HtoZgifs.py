# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:40:43 2023

@author: lukem
"""



import os
import sys
import wget
import zipfile

import time
import random
import numpy as np
import pandas as pd
import imageio
from PIL import Image
from IPython import display

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.image import imread

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from torchsummary import summary

"""
Ignoring FutureWarning
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
import gc
gc.collect()
torch.cuda.empty_cache()


cycleGAN_result_dir =  'CycleGAN_Results/'

def create_and_display_gif(result_dir, start_epoch=149, stop_epoch = 200,  show=True):
    """
    GIF Creation and dispaly conditionally
    """
    images = []
    for epoch in range(start_epoch,stop_epoch):
        file_path = result_dir + 'CycleGAN_Result_Epoch_{:d}'.format(epoch + 1) + '.png'
        images.append(imageio.imread(file_path))
    
    """
    GIF Creation
    """
    gif_file_name = f'CycleGAN_Train_GIF_From_{start_epoch}_to_{stop_epoch}_Epochs.gif'
    
    imageio.mimsave(result_dir + gif_file_name, images)
    print('GIF File : ',gif_file_name, ' is created at ', result_dir)
    
    """
    Display GIF
    """
    if show:
        with open(result_dir + gif_file_name,'rb') as f:
            display.display(display.Image(data=f.read(), format='png'))
            
            



"""
GIF of Train Result Creation and Display.(From epoch 0 to 199)
"""
create_and_display_gif(result_dir=cycleGAN_result_dir, start_epoch=0, stop_epoch = 200, show=False)

