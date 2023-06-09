import torch
import torch.nn as nn
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------------
# Discriminator according to the paper: https://arxiv.org/pdf/1703.10593.pdf
# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Jun-Yan Zhu et.al. 2017/2020)
# Use 70 × 70 PatchGAN. 
# Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. 
# After the last layer, apply a convolution to produce a 1-dimensional output. 
# Do not use InstanceNorm for the first C64 layer. 
# Use leaky ReLUs with a slope of 0.2. 
# The discriminator architecture is: C64-C128-C256-C512
#-----------------------------------------------------------------------------------------------------------------------------

class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            return x + torch.empty_like(x).normal_(std=self.std).to(device)
        else:
            return x




class Discriminator(torch.nn.Module):
    '''
    Discriminator network for PatchGAN
    Outputs (n x m) where each value represents the probability that the patch is real. Each pixel in the output represents a 70 by 70 patch in the input image.
    These patches overlap. 
    '''
    def __init__(self, in_channels=3, num_filters=64, num_layers=3,std=0.1, std_decay_rate=0):
        super(Discriminator, self).__init__()
        

        layers = []

        # C64
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1, padding_mode="reflect"))
        layers += [nn.LeakyReLU(0.2)]

        # C128-C256-C512
        for _ in range(num_layers-1):
            #layers.append(GaussianNoise(std,std_decay_rate))
            layers.append(nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, padding_mode="reflect"))
            layers += [nn.InstanceNorm2d(num_filters*2)]            
            layers += [nn.LeakyReLU(0.2)]
            #layers += [nn.Dropout(0.3)]
            num_filters *= 2

        # output 1 channel prediction map
        layers.append(nn.Conv2d(num_filters, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        self.discriminator = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        
        #print(x.shape)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #x = x + (10**0.5)*torch.randn(x.shape[0], x.shape[1], x.shape[2],x.shape[3]).to(device)
        #discrim = 
        
        return self.discriminator(x)