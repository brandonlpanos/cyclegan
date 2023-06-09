# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:13:05 2023

@author: lukem
"""
import torch.nn as nn

def activation_func(activation_name):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky', nn.LeakyReLU(0.2, inplace=True)],
        ['tanh', nn.Tanh()],
        ['none', nn.Identity()]
    ])[activation_name]

pad_func=lambda kernel_size: (kernel_size-1)//2



# In[13]:


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padded=False, activation='relu', norm=True):
        super().__init__()

        kernel = (kernel_size,kernel_size)
        """
        if Reflection pad is used, set padding param to 0 as already padded 
        """
        padding = pad_func(kernel_size) if not padded else 0 

        self.conv = nn.Conv2d(in_channels,out_channels,kernel,stride,padding)
        self.norm = norm
        self.ins = nn.InstanceNorm2d(out_channels)
        self.activation = activation_func(activation)
        

    def forward(self,x):

        if self.norm:
            x = self.ins(self.conv(x))
        else:
            x = self.conv(x)

        return self.activation(x)



# In[14]:


class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()

        pad = pad_func(kernel_size)
        out_pad=pad
        kernel = (kernel_size,kernel_size)

        self.deconv = nn.ConvTranspose2d(in_channels,out_channels,kernel,stride,pad,out_pad)
        self.ins = nn.InstanceNorm2d(out_channels)
        self.relu = activation_func('relu')

    def forward(self,x):
            x = self.relu(self.ins(self.deconv(x)))
            return x



# In[15]:


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        """
        Input and channel remain same (i.e. 256 ==> R256 as per paper.)
        Keeping stride = 1 to maintain the shape.This two also eleminate Shortcut part to make 1x1 convolution as a "projection". 
        """
        """
        128*64*64 To 128*66*66
        """
        pad=pad_func(kernel_size)
        self.reflection_pad = nn.ReflectionPad2d(pad)
        """
        128*64*64 To 128*64*64  
        then reflection_pad so 128*64*64 To 128*66*66
        """
        self.conv1 = Conv(channels,channels,kernel_size,stride=stride,padded=True)
        """
        128*66*66 To 128*64*64
        """
        self.conv2 = Conv(channels,channels,kernel_size,stride=stride,padded=True,activation='none')
      
        self.relu1 = activation_func('relu')

        """
        Shortcut part is the identify function, which returns the input as the output
        Unless the output of will have a different shape due to a change in
        the number of channels or stride, then we will make the short cut
        a 1x1 convolution as a "projection" to change it's shape.
         
        Which in this case will never execute as channels are same and stride=1. Hence skiping that part.
        """ 
          

    def forward(self,x):
        """
        Compute the results of F_x and x, as needed 
        """
        residual=x
        f_x = self.conv1(self.reflection_pad(x))
        f_x = self.conv2(self.reflection_pad(f_x))
        x = self.relu1(residual + f_x)
        return x

