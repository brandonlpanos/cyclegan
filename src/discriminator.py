import torch
import torch.nn as nn

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
#DOES THIS WORK!!!!!!?