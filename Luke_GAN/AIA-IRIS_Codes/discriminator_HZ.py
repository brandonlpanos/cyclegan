# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:22:30 2023

@author: lukem
"""

from NetworkBlocks import Conv
import torch.nn as nn
import config_HZ as config


class Discriminator(nn.Module):
    def __init__(self,in_channels,n_filter,out_channels,kernel_size=config.patchsize):
        super().__init__()
        """
        C64
        3*256*256 To 64*128*128
        """
        discriminator = nn.ModuleList([Conv(in_channels,n_filter,kernel_size=kernel_size,stride=2,activation='leaky',norm=False)])
        """
        C128
        64*128*128 To 128*64*64
        """
        discriminator += nn.ModuleList([Conv(n_filter,n_filter*2,kernel_size=kernel_size,stride=2,activation='leaky')])
        """
        C256
        128*64*64 To 256*32*32
        """
        discriminator += nn.ModuleList([Conv(n_filter*2,n_filter*4,kernel_size=kernel_size,stride=2,activation='leaky')])
        """
        C512
        256*32*32 To  512*31*31
        """
        discriminator += nn.ModuleList([Conv(n_filter*4,n_filter*8,kernel_size=kernel_size,stride=1,activation='leaky')])
        """
        Final layer, so no need of normalization and activation.
        512*31*31 To  1*30*30
        """
        discriminator += nn.ModuleList([Conv(n_filter*8,out_channels,kernel_size=kernel_size,stride=1,activation='none',norm=False)])

        
        self.discriminator =nn.Sequential(*discriminator)
      
    def forward(self,x):
        x = self.discriminator(x)
        return x
