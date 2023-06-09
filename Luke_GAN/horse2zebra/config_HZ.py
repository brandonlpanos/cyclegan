# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:45:36 2023

@author: lukem
"""
import torch
from torchvision import transforms




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

home_dir = 'C:/Users/lukem/OneDrive - Northumbria University - Production Azure AD/Documents/PhD Research/Bern/GAN/horse2zebra'

data_dir='horse2zebra'

"""
Epochs
"""
epochs=200
decay_epoch=100
epoch_offset=1
"""
Size of feature maps in generator. Set the value as per DCGAN.
"""
ngf=64 
"""
Size of feature maps in discriminator.Set the value as per DCGAN.
""" 
ndf=64
"""
Number of residual blocks
""" 
num_residual_blocks=9

"""
Generator learning rate  
""" 
lr_G=0.0002
"""
Discriminator learning rate
""" 
lr_D=0.0002


transform = {
             'train': transforms.Compose([transforms.Resize(size=286),
                                          transforms.CenterCrop(256),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
             'test': transforms.Compose([transforms.Resize(size=256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
             }