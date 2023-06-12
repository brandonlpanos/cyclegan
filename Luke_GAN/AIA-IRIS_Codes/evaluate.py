# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:38:42 2023

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
from dataset_Luke_diffsizes import AIAIRISDataset
from generator_HZ import Generator
from discriminator_HZ import Discriminator
import config_HZ as config

print("a")

#import New_training
home_dir = config.home_dir
os.chdir(home_dir)

results_dir = config.result_dir
cycleGAN_checkpoint_dir = results_dir +  'CycleGAN_Checkpoint/'
cycleGAN_test_resut_x2y2x_dir=results_dir + 'CycleGAN_Test_Results/XtoYtoX/'
cycleGAN_test_resut_y2x2y_dir=results_dir + 'CycleGAN_Test_Results/YtoXtoY/'


device = config.device
#print(device)

data_dir=config.data_dir

"""
Epochs
"""
epochs=config.epochs
decay_epoch=config.decay_epoch
epoch_offset=config.epoch_offset
"""
Size of feature maps in generator. Set the value as per DCGAN.
"""
ngf=config.ngf
"""
Size of feature maps in discriminator.Set the value as per DCGAN.
""" 
ndf=config.ndf
"""
Number of residual blocks
""" 
num_residual_blocks=config.num_residual_blocks

"""
Generator learning rate  
""" 
lr_G=config.lr_G
"""
Discriminator learning rate
""" 
lr_D=config.lr_D

in_channels = config.in_channels
transform = config.transform



checkpoint_dict = torch.load(cycleGAN_checkpoint_dir + 'CycleGAN.pt')
loaded_results_df =  pd.DataFrame.from_dict(checkpoint_dict['results'])


plt.figure(figsize=(10,5))
plt.title("Generators and Discriminators Losses and Cyclic Losses During CycleGAN Training", fontsize=16)
plt.xlabel('Number of Epochs', fontsize=14)
plt.ylabel('Train Losses', fontsize=14)

for col in loaded_results_df.columns[2:]:
    plt.plot(loaded_results_df[col], label=col)

plt.legend()
plt.show()


def create_cyclegan_model(n_gen_filter, n_dcrmnt_filter, n_residual_blocks, load_state=False,in_channels = in_channels):
    """
    * Creates 2 Generators and 2 Discriminators.
    * In case of restoring the states of original models this function will only create 2 Generators.
    * Place the created models on the correct compute resource (CPU or GPU).
    * Models' weight initialized from a Gaussian distribution N (0, 0.02) except for restoring the states of original models.
    """
    
    """
    Create Generators
    """
    G_XtoY = Generator(in_channels=in_channels, n_filter=n_gen_filter, out_channels=in_channels, n_residual_blocks=n_residual_blocks)
    G_YtoX = Generator(in_channels=in_channels, n_filter=n_gen_filter, out_channels=in_channels, n_residual_blocks=n_residual_blocks)

    """
    Place the models on the correct compute resource (CPU or GPU)
    """
    G_XtoY.to(device)
    G_YtoX.to(device)

    print('Created Generators and move them to the correct compute resource (CPU or GPU)')
   
    """
    Create Discriminators and Place the models on the correct compute resource (CPU or GPU).
    Models' weight initialized from a Gaussian distribution N (0, 0.02)
    """
    if not load_state:
        G_XtoY.apply(weights_init)
        G_YtoX.apply(weights_init)

        print('Generators\' weight initialized from a Gaussian distribution N (0, 0.02)')

        D_X = Discriminator(in_channels=in_channels,n_filter=n_dcrmnt_filter,out_channels=1)
        D_Y = Discriminator(in_channels=in_channels,n_filter=n_dcrmnt_filter,out_channels=1)

        D_X.to(device)
        D_Y.to(device)
        
        print('Created Discriminators and move them to the correct compute resource (CPU or GPU)')
        
        D_X.apply(weights_init)
        D_Y.apply(weights_init)

        print('Discriminators\' weight initialized from a Gaussian distribution N (0, 0.02)')


    if not load_state:
        return G_XtoY, G_YtoX, D_X, D_Y
    else:
        return G_XtoY, G_YtoX
    
    
def weights_init(m):
    for layer in m.children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)



# In[19]:


def create_cyclegan_model(n_gen_filter, n_dcrmnt_filter, n_residual_blocks, load_state=False,in_channels = in_channels):
    """
    * Creates 2 Generators and 2 Discriminators.
    * In case of restoring the states of original models this function will only create 2 Generators.
    * Place the created models on the correct compute resource (CPU or GPU).
    * Models' weight initialized from a Gaussian distribution N (0, 0.02) except for restoring the states of original models.
    """
    
    """
    Create Generators
    """
    G_XtoY = Generator(in_channels=in_channels, n_filter=n_gen_filter, out_channels=in_channels, n_residual_blocks=n_residual_blocks)
    G_YtoX = Generator(in_channels=in_channels, n_filter=n_gen_filter, out_channels=in_channels, n_residual_blocks=n_residual_blocks)

    """
    Place the models on the correct compute resource (CPU or GPU)
    """
    G_XtoY.to(device)
    G_YtoX.to(device)

    print('Created Generators and move them to the correct compute resource (CPU or GPU)')
   
    """
    Create Discriminators and Place the models on the correct compute resource (CPU or GPU).
    Models' weight initialized from a Gaussian distribution N (0, 0.02)
    """
    if not load_state:
        G_XtoY.apply(weights_init)
        G_YtoX.apply(weights_init)

        print('Generators\' weight initialized from a Gaussian distribution N (0, 0.02)')

        D_X = Discriminator(in_channels=in_channels,n_filter=n_dcrmnt_filter,out_channels=1)
        D_Y = Discriminator(in_channels=in_channels,n_filter=n_dcrmnt_filter,out_channels=1)

        D_X.to(device)
        D_Y.to(device)
        
        print('Created Discriminators and move them to the correct compute resource (CPU or GPU)')
        
        D_X.apply(weights_init)
        D_Y.apply(weights_init)

        print('Discriminators\' weight initialized from a Gaussian distribution N (0, 0.02)')


    if not load_state:
        return G_XtoY, G_YtoX, D_X, D_Y
    else:
        return G_XtoY, G_YtoX


def real_gen_recon_image(G_1,G_2,real_image):
    """
    This function is used to generate fake and reconstructed images generated by generators
    """
    """
    Move image to the device.
    """
    real_image = real_image.to(device)

    """
    Real To Genereted To Reconstruction
    """
    fake_image = G_1(real_image)
    reconstructed_image = G_2(fake_image)

    return fake_image,reconstructed_image



def to_numpy_and_scale(x):
   """
   Function to prepare the image tensor to work with matplotlib 
   """
   grid = torchvision.utils.make_grid(x.clamp(min=-1, max=1), scale_each=True, normalize=True)
   
   return grid.permute(1, 2, 0).detach().cpu().numpy()

def generate_result(real_image, gen_image, recon_image, epoch, result_dir, is_test=False, show=False):
    """
    Create and conditinaly show real image with fake and reconstructed images generated by generators.
    This function is used to generate both train and test result based on parameters.
    """
    titles = ['Real', 'Generated', 'Reconstructed']
    if is_test:
        images=[to_numpy_and_scale(real_image[0]),  to_numpy_and_scale(gen_image[0]), to_numpy_and_scale(recon_image[0])]
        fig, axarr = plt.subplots(1, 3, figsize=(10,10))
    else:
        images = [to_numpy_and_scale(real_image[0]), to_numpy_and_scale(gen_image[0]), to_numpy_and_scale(recon_image[0]),
                  to_numpy_and_scale(real_image[1]), to_numpy_and_scale(gen_image[1]), to_numpy_and_scale(recon_image[1])]
                    
        fig, axarr = plt.subplots(2, 3, figsize=(10,10))

    for i in range(len(images)):
        if not is_test:
            if i < 3:
                nrows=0
                ncols=i
                
                title_i=i
            else:
                nrows=1
                ncols=i - 3
                title_i=i-3
            ax=axarr[nrows][ncols]
        else:
            title_i=i
            ax=axarr[i]
            
  
        """
        Turn off axis of the plot
        """
        ax.set_axis_off()
        """
        Plot image data
        """
        
        ax.imshow(images[i], aspect='equal')
        """
        Set Title of individual subplot
        """
        ax.set_title(titles[title_i], color='red', fontsize = 16)
    """
    Tune the subplot layout
    """
    plt.subplots_adjust(wspace=0, hspace=0)

    if not is_test:
        """
        Add the text for train and validation image.
        Add the text to the axes at location coordinates.
        """
        fig.text(0.5, 0.05, 'Epoch {}'.format(epoch + 1), horizontalalignment='center', fontsize=16, color='red')

    """
    Save every plot.
    """
    if not is_test:
        result_file = os.path.join(result_dir,'CycleGAN_Result_Epoch_{}'.format(epoch+1) + '.png')
    else:    
        result_file = os.path.join(result_dir + 'CycleGAN_Test_Result_{}'.format(epoch + 1) + '.png')

    plt.savefig(result_file)

    """
    Display(Conditional)
    """
    if show and is_test:
         plt.show()
    else:
        plt.close()



















G_XtoY, G_YtoX = create_cyclegan_model(n_gen_filter=config.ngf, n_dcrmnt_filter=config.ndf, n_residual_blocks=config.num_residual_blocks, 
                                       load_state=True)

G_XtoY.load_state_dict(checkpoint_dict['G_XtoY_state_dict'])

G_YtoX.load_state_dict(checkpoint_dict['G_YtoX_state_dict'])

G_XtoY = G_XtoY.eval()
G_YtoX = G_YtoX.eval()

"""
"""

test_data_X = AIAIRISDataset(image_dir=data_dir, is_train=False, image_type='aia', transform=transform)
                        
test_loader_X = DataLoader(dataset=test_data_X, batch_size=1, shuffle=False)

test_data_Y = AIAIRISDataset(image_dir=data_dir, is_train=False, image_type='iris', transform=transform)
                        
test_loader_Y = DataLoader(dataset=test_data_Y, batch_size=1, shuffle=False)



print('AIA ----------------> IRIS ----------------> AIA\n\n')
for i, real_X in enumerate(tqdm(test_loader_X, desc="Test Batch X To Y To X", leave=False, disable=False)):
    """
    X To Y To X
    """
    fake_Y, reconstructed_X = real_gen_recon_image(G_XtoY, G_YtoX, real_X)
  
    """
    Generating result for all test data of domain X
    Showing only few results
    """
    show=False
    if i in [2,8,11,49,78,88,101,111]:
        show=True

    generate_result([real_X], [fake_Y], [reconstructed_X], i, result_dir=cycleGAN_test_resut_x2y2x_dir, is_test=True, show=show)

print('\n%d test images (X To Y To X) are generated.\n\n' % (i + 1))

print('IRIS ----------------> AIA ----------------> IRIS\n\n')
for i, real_Y in enumerate(tqdm(test_loader_Y, desc="Test Batch Y To X To Y", leave=False, disable=False)):
    """
    Y To X To Y
    """
    fake_X, reconstructed_Y = real_gen_recon_image(G_YtoX, G_XtoY, real_Y)

    """
    Generating result for all test data  of domain Y
    Showing only few results
    """
    show=False
    if i in [27,28,58,67,91,93]:
        show=True
        
    generate_result([real_Y], [fake_X], [reconstructed_Y], i, result_dir=cycleGAN_test_resut_y2x2y_dir, is_test=True, show=show)

print('\n%d test images (Y To X To Y) are generated.' % (i + 1))