#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:49:46 2023

@author: w22038792
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import sunkit_image.enhance as enhance


data_directory = "/home/w22038792/Documents/Bern/GAN_data/"
save_directory = "/home/w22038792/Documents/GitHub/ToyCycleGAN/aiairis_cyclegan/data/difsizes/"


def check_if_dir_exists(directory):
    
    return(os.path.isdir(directory))

def make_dir(directory):
    
    if not check_if_dir_exists(directory):
        os.mkdir(directory)
        
        
make_dir(save_directory + "aia_train")
make_dir(save_directory + "aia_test")
make_dir(save_directory + "iris_train")
make_dir(save_directory + "iris_test")



obs_list = os.listdir(data_directory +"sdo/")

train_set, test_set = train_test_split(obs_list, test_size=0.2, random_state=42)
train_obs_list = train_set
test_obs_list = test_set


def crop_center(img,cropx,cropy):
    print("start ",img.shape)
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def paths(obs_list,size,ttype):

    iris_channel = '1400'
    sdo_channels = ['304']
    
    iris_paths = []
    for obs in obs_list:
        path_to_iris_obs = data_directory +f'iris/{obs}/{iris_channel}/'
        for file in os.listdir(path_to_iris_obs):
            iris_paths.append(f'{path_to_iris_obs}/{file}')
    
    # Collect paths to SDO images
    sdo_paths = []
    for obs in obs_list:
        path_to_sdo_obs = data_directory +f'sdo/{obs}/'
        for sdo_channel in sdo_channels:
            path_to_sdo_obs_channel = f'{path_to_sdo_obs}/{sdo_channel}/'
            for file in os.listdir(path_to_sdo_obs_channel):
                sdo_paths.append(f'{path_to_sdo_obs_channel}/{file}')
                
    
                
    print(len(iris_paths))
    iris_list = np.random.choice(iris_paths, size = size, replace = False)
    sdo_list = np.random.choice(sdo_paths, size = size, replace = False)
    
    
    
    
    irisImages = []
    t = 0
    for i in iris_list:
      data = np.load(i)
      
      
      np.save(save_directory+ "iris_"+ttype+str(t)+".npy", data)

      #data = crop_center(data, 128, 128)
      print(data.shape)
      #data = np.array(Image.fromarray(data.astype(np.uint8)).resize((512, 512))).astype('float32')
      #data = np.resize(data, (255,255))
      t+=1
      #irisImages.append(data)        
    
    #np.save("/home/w22038792/Documents/GitHub/ToyCycleGAN/aiairis_cyclegan/data/difsizes/iris_"+ttype+".npy", irisImages)
    
    
    sdoImages = []
    t = 0
    for i in sdo_list:
      data = np.load(i)
      np.save(save_directory+ "aia_"+ttype+str(t)+".npy", data)

     # print(data.shape)
      #data = crop_center(data, 128, 128)
      #data = np.array(Image.fromarray(data.astype(np.uint8)).resize((256, 256))).astype('float32')
      #plt.imshow(data)
      #plt.show()
      t+=1
     # sdoImages.append(data)        
    
    #np.save("/home/w22038792/Documents/GitHub/ToyCycleGAN/aiairis_cyclegan/data/difsizes/aia_"+ttype+".npy", irisImages)
    
paths(train_obs_list,1800,"train/")
paths(test_obs_list,200,"test/")

    



