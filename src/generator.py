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


class Generator(nn.Module):
    '''
    Generator network with 9 residual blocks
    '''

    def __init__(self, num_res_blocks=9):
        super(Generator, self).__init__()
        # c7s1-64
        self.generator = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=1),
            # d128
            ConvBlock(64, 128, kernel_size=3, stride=2),
            # d256
            ConvBlock(128, 256, kernel_size=3, stride=2),
            # R256
            *[ResBlock(256, 256) for _ in range(num_res_blocks)],
            # u128
            FractionalStridedConvBlock(256, 128),
            # u64
            FractionalStridedConvBlock(128, 64),
            # c7s1-3
            ConvBlock(64, 3, kernel_size=7, stride=1,
                      norm=False, activation='Tanh')
        )

    def forward(self, x):
        '''Forward pass'''
        return self.generator(x)


# c7s1 #dk
class ConvBlock(nn.Module):
    '''
    Convolutional block with optional normalization and activation
    '''

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, norm=True, activation='ReLU'):
        super(ConvBlock, self).__init__()
        layers = []
        # Add reflection padding to reduce artifacts
        layers += [nn.ReflectionPad2d(3)]
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers += [select_activation(activation)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        return self.conv_block(x)


# Rk
class ResBlock(nn.Module):
    '''
    Residual block with optional normalization and activation
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=True, activation='ReLU'):
        super(ResBlock, self).__init__()
        layers = []
        # Add reflection padding to reduce artifacts
        layers += [nn.ReflectionPad2d(1)]
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers += [select_activation(activation)]
        layers += [nn.ReflectionPad2d(1)]
        layers.append(nn.Conv2d(out_channels, out_channels,
                      kernel_size, stride=1))  # Set stride to 1
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        return x + self.res_block(x)


# Uk
class FractionalStridedConvBlock(nn.Module):
    '''
    Fractional strided convolutional block with optional normalization and activation. Increases the spatial resolution by a factor of 2
    '''

    def __init__(self, in_channels, out_channels):
        super(FractionalStridedConvBlock, self).__init__()
        layers = []
        layers += [nn.ConvTranspose2d(in_channels, out_channels,
                                      3, stride=2, padding=1, output_padding=1)]
        layers += [nn.InstanceNorm2d(out_channels)]
        layers += [nn.ReLU()]
        self.fractional_block = nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        return self.fractional_block(x)


# Choose between ReLU and LeakyReLU
def select_activation(activ='ReLU'):
    '''
    Choose between ReLU and LeakyReLU
    '''
    if activ == 'LeakyReLU':
        activation = nn.LeakyReLU()
    else:
        return nn.ReLU()
    return activation