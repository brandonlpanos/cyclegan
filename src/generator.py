import torch.nn as nn

# -----------------------------------------------------------------------------------------------------------------------------
# Generator according to the paper: https://arxiv.org/pdf/1703.10593.pdf
# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Jun-Yan Zhu et.al. 2017/2020)

# Use 9 resblocks
# c7s1-k denotes a 7×7 Convolution-InstanceNorm-ReLU with k filters and stride 1
# dk denotes a 3×3 Convolution-InstanceNorm-ReLU with k filters and stride 2
# Reflection padding was used to reduce artifacts
# Rk denotes a residual block that contains two 3×3 convolutional layers with the same number of filters on both layer
# uk denotes a 3×3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2
# Use reflection padding to reduce artifacts:
# for Conv nn.ReflectionPad2d(3)
# for Res nn.ReflectionPad2d(1)
# The generator archetecture is: c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
# --------------------------------------------------------------------------------------------------------------------------------