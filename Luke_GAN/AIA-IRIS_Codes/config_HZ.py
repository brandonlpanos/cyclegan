# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:45:36 2023

@author: lukem
"""
import torch
from torchvision import transforms




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)

home_dir = '/home/osw_w22038792/Bern'

data_dir='1800_data/'

result_dir = 'Results1800_NAM/'

"""
Epochs
"""
epochs=100
decay_epoch=80
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
num_residual_blocks=6

"""
Generator learning rate  
""" 
lr_G=0.00002
"""
Discriminator learning rate
""" 
lr_D=0.00002

in_channels = 1



transform = {
             'iris': transforms.Compose([transforms.Resize(size=286),
                                          transforms.CenterCrop(256),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
             'sdo': transforms.Compose([transforms.Resize(size=256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
}



tilegridsize = (8,8)

patchsize = 4

cycleloss = 7

             
