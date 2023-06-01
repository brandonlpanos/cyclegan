#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:09:18 2023

@author: w22038792
"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            #nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3,stride =1, padding_mode="reflect",padding=1),
            nn.InstanceNorm2d(in_features), 
            nn.ReLU(inplace=True),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3,stride =1,  padding_mode="reflect", padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(Generator, self).__init__()
        
        self.input_shape = input_shape
        channels = self.input_shape[0]
        
        # Initial Convolution Block
        out_features = 64
        model = [
            #nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, kernel_size=7, stride = 1, padding_mode="reflect",padding = 3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding_mode="reflect",padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]
            
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                #nn.Upsample(scale_factor=2), # --> width*2, heigh*2
                #nn.Conv2d(in_features, out_features, 2, stride=1, padding=1),
                nn.ConvTranspose2d(in_features, out_features, kernel_size = 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            
        # Output Layer
        #print(out_features)
        model += [#nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, kernel_size=7,stride = 1, padding_mode="reflect",padding = 3),
                  nn.Tanh()
                 ]
        
        # Unpacking
        self.model = nn.Sequential(*model) 
        
    def forward(self, x):
        x = self.model(x)
        # Resize to input shape. 
        # Sometimes the output of the generator is not the same size as the input by a few pixels.
        # We assume it is fine to resize the output. F.interpolate is part of the computational graph.
        if x.shape != self.input_shape:
            x = F.interpolate(x, size=self.input_shape[2:], mode='bilinear', align_corners=False)
        return x