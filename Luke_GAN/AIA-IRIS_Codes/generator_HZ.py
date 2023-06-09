# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:15:40 2023

@author: lukem
"""

from NetworkBlocks import Conv,Deconv,ResidualBlock
import torch.nn as nn

pad_func=lambda kernel_size: (kernel_size-1)//2


class Generator(nn.Module):
    def __init__(self, in_channels, n_filter, out_channels, n_residual_blocks,kernel_size=7):
        super().__init__()
        """
        Component of generator : 
            * Initial Convolution Block
            * Encoder
            * Residual blocks
            * Decoder
            * Output Convolution Block

        kernel_size=7 for two conv layers : Initial Convolution Block and Output Convolution Block.
        But rest conv layers of encoder and residual block or deconv layers of decoder have 3 as kernal size which is by defalut initialzed
        by the Conv and Deconv class.
        """

        """
        Initial Convolution Block
        Reflection padding ==> 3*256*256 To 3*262*262
        c7s1-64 ==>#3*262*262 To 64*256*256

        Generator input size is  3 * 256 * 256
        As per paper, this initial conv layer will have kernel size=7 so inorder to keep the image size (W,H) same 
        we need to pad it by padding of size (kernel_size-1)//2 =7-1//2 = 3
        As per paper I use Reflection padding to reduce artifact.
        """
        pad = pad_func(kernel_size)
        generator = nn.ModuleList([nn.ReflectionPad2d(pad), #3*256*256 To 3*262*262
                     Conv(in_channels,n_filter,kernel_size=kernel_size,stride=1,padded=True) #3*262*262 To 64*256*256
                    ])
      
        """
        Encoder
        Downsampling
        d128 ==> 64*256*256 To 128*128*128
        d256 ==> 128*128*128 To 256*64*64
        """
        generator += nn.ModuleList([Conv(n_filter,n_filter*2), #64*256*256 To 128*128*128
                      Conv(n_filter*2,n_filter*4)#128*128*128 To 256*64*64
                     ])

        """
        Residual blocks : R256,R256,R256,R256,R256,R256,R256,R256,R256
        ==> 256*64*64 To 256*64*64
        """
      
        generator +=nn.ModuleList([ResidualBlock(n_filter*4) for i in range(n_residual_blocks)])#256*64*64 To 256*64*64
        
        """
        Decoder
        Upsampling
        u128 ==> 256*64*64 To 128*128*128
        u64 ==> #128*128*128 To 64*256*256 
        """
        generator += nn.ModuleList([Deconv(n_filter*4,n_filter*2),#256*64*64 To 128*128*128
                      Deconv(n_filter*2,n_filter)#128*128*128 To 64*256*256 Then reflection_pad so 64*256*256 To 64*262*262
                     ])
        
        """
        Output Layer
        Then reflection_pad so 64*256*256 To 64*262*262
        c7s1-3 ==> 64*262*262 To 3*256*256 

        The previous decoder gives image outcome of size 64*256*256.
        Discriminator takes image of size 3*256*256
        As per paper, this output conv layer will have kernel size=7 
        so inorder to keep the image size (W,H) same 
        need to pad it by padding of size (kernel_size-1)//2 =7-1//2 = 3
        As per paper I use Reflection padding to reduce artifact.
        """
        generator += nn.ModuleList([nn.ReflectionPad2d(pad),
                      Conv(n_filter,out_channels,kernel_size=kernel_size,stride=1,padded=True,activation='tanh',norm=False)#64*262*262 To 3*256*256
                     ])
        
        self.generator = nn.Sequential(*generator)

    def forward(self,x):
        
        
        fake = self.generator(x)
        
        #print(fake.shape)
        realshape = x.shape
        fake_shape = (fake.shape[1],fake.shape[2],fake.shape[3])
        #fake_shape = fake.shape
        
        real_shape = realshape
        #real_shape = (realshape[1],realshape[2],realshape[3])
        
        #print(fake_shape,real_shape)
        
        
        
        #print("Real: ",realshape)
        
        #print(fake_shape,real_shape)
        #print(fake.shape)
        
        
        
        if fake_shape != real_shape:
            fake = nn.functional.interpolate(fake, size=(real_shape[2],real_shape[3]), mode='bicubic', align_corners=False)
        
        return fake
