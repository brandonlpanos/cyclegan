#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install wget

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

"""
Ignoring FutureWarning
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
import gc
gc.collect()
torch.cuda.empty_cache()

# In[2]:


"""
Required Functions For directory Creation
"""
def check_if_dir_exists(directory):
    """
    Checks if 'directory' exists
    """
    return(os.path.isdir(directory))

def make_dir(directory):
    """
    Crete directory
    """
    if not check_if_dir_exists(directory):
        os.mkdir(directory)
        print("Directory %s created successfully." %directory)
    else:
        print("Directory %s exists." %directory)

"""
Required directory Creation
"""
"""cycleGAN_dir='/content/gdrive/My Drive/DATA690_Project_CYCLEGAN'
make_dir(cycleGAN_dir)

os.chdir('/content/gdrive/My Drive/DATA690_Project_CYCLEGAN')"""

cycleGAN_result_dir =  'CycleGAN_Results/'
make_dir(cycleGAN_result_dir)

cycleGAN_validation_result_dir =  'CycleGAN_Validation_Results/'
make_dir(cycleGAN_validation_result_dir)

cycleGAN_test_resut_dir='CycleGAN_Test_Results/'
make_dir(cycleGAN_test_resut_dir)

cycleGAN_test_resut_x2y2x_dir='CycleGAN_Test_Results/XtoYtoX/'
make_dir(cycleGAN_test_resut_x2y2x_dir)

cycleGAN_test_resut_y2x2y_dir='CycleGAN_Test_Results/YtoXtoY/'
make_dir(cycleGAN_test_resut_y2x2y_dir)


cycleGAN_checkpoint_dir =  'CycleGAN_Checkpoint/'
make_dir(cycleGAN_checkpoint_dir)



# In[3]:


"""
Device
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_dir='horse2zebra'

"""
Epochs
"""
epochs=200
decay_epoch=100
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
num_residual_blocks=9

"""
Generator learning rate  
""" 
lr_G=0.0002
"""
Discriminator learning rate
""" 
lr_D=0.0002


# In[4]:


class ImageDataset(Dataset):
    def __init__(self, image_dir, is_train, image_type):
        self.train_or_test='train' if is_train else 'test'
        self.image_dir = './' + image_dir
        self.image_type=image_type
        self.image_path = os.path.join(self.image_dir, self.train_or_test+'{}'.format(self.image_type))
        self.image_filename_lst = [x for x in sorted(os.listdir(self.image_path))]
        self.transform = transform[self.train_or_test]
        

    def __getitem__(self, index):
        image_file = os.path.join(self.image_path, self.image_filename_lst[index])
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)
        return image
 
    def __len__(self):
        return len(self.image_filename_lst)


# In[5]:


transform = {
             'train': transforms.Compose([transforms.Resize(size=286),
                                          transforms.CenterCrop(256),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
             'test': transforms.Compose([transforms.Resize(size=256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            }
"""
Train Data Loader 
"""
train_data_X = ImageDataset(image_dir=data_dir, is_train=True, image_type='A')
                        
train_loader_X = DataLoader(dataset=train_data_X, batch_size=1, shuffle=True)

train_data_Y = ImageDataset(image_dir=data_dir, is_train=True, image_type='B')
                        
train_loader_Y = DataLoader(dataset=train_data_Y, batch_size=1, shuffle=True)

"""
Test Data Loader
"""
test_data_X = ImageDataset(image_dir=data_dir, is_train=False, image_type='A')
                        
test_loader_X = DataLoader(dataset=test_data_X, batch_size=1, shuffle=False)

test_data_Y = ImageDataset(image_dir=data_dir, is_train=False, image_type='B')
                        
test_loader_Y = DataLoader(dataset=test_data_Y, batch_size=1, shuffle=False)


""# In[6]:


train_real_H = train_data_X.__getitem__(26)
train_real_Z = train_data_Y.__getitem__(91)

val_real_H = test_data_X.__getitem__(78)
val_real_Z = test_data_Y.__getitem__(67)


# In[7]:





# In[9]:


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

img_idx_lst=np.random.randint(0,1000,8)

def show_images(data_X, data_Y):
    rows, cols,=2, 4
    f, axarr = plt.subplots(rows,cols, figsize=(20,10))

    for i in range(rows):
        for j in range(cols):
            if i==0:
                x = data_X.__getitem__(img_idx_lst[i*4+j])
                s='HORSE'
            else :
                x = data_Y.__getitem__(img_idx_lst[i*4+j])
                s='ZEBRA'

            grid = torchvision.utils.make_grid(x.clamp(min=-1, max=1), scale_each=True, normalize=True)
            """
            Turn off axis
            """
            axarr[i,j].set_axis_off()
            """
            Plot image data
            """
            axarr[i,j].imshow(grid.permute(1, 2, 0).cpu().numpy())

            """
            Add the text for validation image.
            Add the text to the axes at location coordinates.
            """
            axarr[i,j].text(0.5, 0.05, s, dict(size=20, color='blue'))
    

show_images(train_data_X, train_data_Y)



# In[10]:


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



# In[11]:


def update_image_buffer_and_get_image(image_buffer, input_images, capacity):

    if capacity == 0:
        return input_images

    return_images = []

    for input_image in input_images.data:
        input_image = torch.stack([input_image])
        """
        Populate the image buffer one by one until its reaches the capacity.
        """
        if len(image_buffer) < capacity:
            image_buffer.append(input_image)
            return_images.append(input_image)

        elif random.random() > 0.5:
            """
            Probabilistically, replace an existing fake image and use replaced fake image.
            """
            randId = random.randint(0, capacity-1)
            return_images.append(image_buffer[randId])
            image_buffer[randId] = input_image
        else:
            """
            Probabilistically, uses a generated fake image directly.
            """
            return_images.append(input_image)
            
    return_images = torch.cat(return_images, 0)
  
    return return_images



# In[12]:


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



# In[16]:


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
        return self.generator(x)



# In[17]:


class Discriminator(nn.Module):
    def __init__(self,in_channels,n_filter,out_channels,kernel_size=4):
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



# In[18]:


"""
Weight initialization from a Gaussian distribution N (0, 0.02)
""" 
def weights_init(m):
    for layer in m.children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)



# In[19]:


def create_cyclegan_model(n_gen_filter, n_dcrmnt_filter, n_residual_blocks, load_state=False):
    """
    * Creates 2 Generators and 2 Discriminators.
    * In case of restoring the states of original models this function will only create 2 Generators.
    * Place the created models on the correct compute resource (CPU or GPU).
    * Models' weight initialized from a Gaussian distribution N (0, 0.02) except for restoring the states of original models.
    """
    
    """
    Create Generators
    """
    G_XtoY = Generator(in_channels=3, n_filter=n_gen_filter, out_channels=3, n_residual_blocks=n_residual_blocks)
    G_YtoX = Generator(in_channels=3, n_filter=n_gen_filter, out_channels=3, n_residual_blocks=n_residual_blocks)

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

        D_X = Discriminator(in_channels=3,n_filter=n_dcrmnt_filter,out_channels=1)
        D_Y = Discriminator(in_channels=3,n_filter=n_dcrmnt_filter,out_channels=1)

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



# In[20]:


G_XtoY, G_YtoX, D_X, D_Y = create_cyclegan_model(n_gen_filter=ngf, n_dcrmnt_filter=ndf, n_residual_blocks=num_residual_blocks)


# In[21]:


generators_parameters = list(G_XtoY.parameters()) + list(G_YtoX.parameters())
optimizer_G = torch.optim.AdamW(generators_parameters,  lr=lr_G, betas=(0.5, 0.999))
optimizer_D_X = torch.optim.AdamW(D_X.parameters(), lr=lr_D, betas=(0.5, 0.999))
optimizer_D_Y = torch.optim.AdamW(D_Y.parameters(), lr=lr_D, betas=(0.5, 0.999))


"""
Loss Functions
"""
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()

"""
Establish convention for real and fake labels during training
"""
real_label = 1.0
fake_label = 0.0


# In[22]:


to_track =["Epochs", "Total_time", "D_X_losses", "D_Y_losses", "G_XtoY_losses", "G_YtoX_losses", "cycle_X_losses", "cycle_Y_losses"]
"""
How long have we spent in the training loop?   
"""
total_train_time = 0     
results = {}

"""
Initialize every item with an empty list.
"""
for item in to_track:
    results[item] = []

"""
Learning rate update schedulers.
Adjust Learing rate : Linear decay of learning rate to zero after 100 epochs.
"""
lambda_lr_func = lambda epoch: 1.0 - max(0, epoch + epoch_offset - decay_epoch) / (epochs - decay_epoch)

lr_scheduler_G   = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda = lambda_lr_func)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda = lambda_lr_func)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda = lambda_lr_func)


"""
Creating image buffer of capacity 50 to hold Generated image as per the paper.
"""
buffer_capacity = 50

fake_X_buffer = []
fake_Y_buffer = []


for epoch in tqdm(range(epochs), desc="Epochs", disable=False):
    """
    Put models in training mode.
    """
    G_XtoY = G_XtoY.train()
    G_YtoX = G_YtoX.train()
    D_X = D_X.train()
    D_Y = D_Y.train()

    G_XtoY_running_loss = 0.0
    G_YtoX_running_loss = 0.0
    D_X_running_loss = 0.0
    D_Y_running_loss = 0.0
    cycle_X_running_loss= 0.0
    cycle_Y_running_loss= 0.0
    
    
    
    start = time.time()
    #i = 0
    for real_X, real_Y in tqdm(zip(train_loader_X, train_loader_Y), desc="Train Batch", leave=False, disable=False):
       # print(i)
        #i+=1
        """
        Move the batch to the device we are using. 
        """
        real_X = real_X.to(device)
        real_Y = real_Y.to(device)
        
        """
        ****************************** Train Generators *******************************
        
        ***************************** Train Generator G_XtoY **************************
        """

        """
        Generator: G_XtoY: real_X -> Fake_Y 
        Forward Pass Through Generator : First, generate fake_Y fake images and reconstruct reconstructed_X images.
        """
        """
        PyTorch stores gradients in a mutable data structure. So we need to set it to a clean state before we use it. 
        Otherwise, it will have old information from a previous iteration.
        """
        optimizer_G.zero_grad()
        """
        1. G_XtoY Generator generates fake_Y fake images that look like domain Y based on real real_X images of domain X.
        """
        fake_Y = G_XtoY(real_X)
        """
        2. Compute the generator loss based on the response of D_Y.
        """
        D_Y_fake_out = D_Y(fake_Y)#1*1*30*30
        G_XtoY_loss = mse_criterion(D_Y_fake_out, torch.full(D_Y_fake_out.size(), real_label, device=device))
        """
        3. G_YtoX Generator generates reconstructed reconstructed_X images based on the fake_Y fake images generated in step 1.
        """
        reconstructed_X = G_YtoX(fake_Y)
        """
        Forward Cycle Consistency Loss
        Forward cycle loss:  lambda * ||G_YtoX(G_XtoY(X)) - X|| (Equation 2 in the paper)
        4. Compute the cycle consistency loss by comparing the reconstructed reconstructed_X images with real real_X  images of domain X.
           Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation. 
        """
        cycle_X_loss = l1_criterion(reconstructed_X, real_X) * 10.0
        
        """
        ***************************** Train Generator G_YtoX **************************
        Generator: G_YtoX: real_Y -> Fake_X
        Backward Pass Through Generator : Now, generate fake_X fake images and reconstruct reconstructed_Y images.
        """
        """
        5. G_YtoX Generator generates fake_X fake images that look like domain X based on real real_Y images of domain Y.
        """
        fake_X = G_YtoX(real_Y)
        """
        6. Compute the generator loss based on the respondse of D_X.
        """
        D_X_fake_out= D_X(fake_X)
        G_YtoX_loss = mse_criterion(D_X_fake_out, torch.full(D_X_fake_out.size(), real_label, device=device))
        """
        7. G_XtoY Generator generates reconstructed reconstructed_Y images based on the fake_X fake images generated in step 5.
        """
        reconstructed_Y = G_XtoY(fake_X)
        """
        Backward Cycle Consistency Loss
        Backward cycle loss: lambda * ||G_XtoY(G_YtoX(Y)) - Y|| (Equation 2)
        8. Compute the cycle consistency loss by comparing the reconstructed reconstructed_Y images with real real_Y images of domain Y.
           Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation.
        """
        cycle_Y_loss = l1_criterion(reconstructed_Y, real_Y) * 10.0
        
        """
        Finally, Total Generators Loss and Back propagation
        9. Add up all the Generators loss and cyclic loss (Equation 3 of paper.also Equation I the code representation of the equation) and perform backpropagation with optimization.
        """
        G_loss = G_XtoY_loss + G_YtoX_loss + cycle_X_loss + cycle_Y_loss
        """
        ∇_Θ just got computed by this one call!
        """
        G_loss.backward()
        """
        Now we just need to update all the parameters! 
        Θ_{k+1} = Θ_k − η * ∇_Θ ℓ(y_hat, y)
        """
        optimizer_G.step()
        
        G_XtoY_running_loss+=G_XtoY_loss.item()
        G_YtoX_running_loss+=G_YtoX_loss.item()
        
        cycle_X_running_loss+=cycle_X_loss.item()
        cycle_Y_running_loss+=cycle_Y_loss.item()
        
        """
        ***************************** Train Discriminators ****************************

        *************************** Train Discriminator D_X ***************************
        Discriminator: D_X: G_YtoX(Y) vs. X 
        First, real and fake loss of Discriminator D_X .
        """
        """
        PyTorch stores gradients in a mutable data structure. So we need to set it to a clean state before we use it. 
        Otherwise, it will have old information from a previous iteration.
        """
        optimizer_D_X.zero_grad()
        """
        Train D_X with real real_X images of domain X.
        1. Compute D_X_real_loss, the real loss of discriminator D_X on real real_X images of domain X.
        """
        D_X_real_out = D_X(real_X)
        D_X_real_loss = mse_criterion(D_X_real_out, torch.full(D_X_real_out.size(), real_label, device=device))
        """
        Train with fake_X fake image(History of generated images stored in the image buffer).
        2. Get generated fake_X fake image from Image Buffer that look like domain X and based on real images in domain Y.
        """
        fake_X = update_image_buffer_and_get_image(fake_X_buffer,fake_X,buffer_capacity)
        """
        3. Compute D_X_fake_loss, the fake loss for discriminator D_X on fake images generated by generator.
        """
        D_X_fake_out = D_X(fake_X)
        D_X_fake_loss = mse_criterion(D_X_fake_out, torch.full(D_X_fake_out.size(), fake_label, device=device))
        """
        Back propagation
        As per the paper, I multiplied the loss for the discriminator by 0.5 during training, 
        in order to slow down updates to the discriminator relative to the generator model during training.
        4. Compute the total loss for D_X, perform backpropagation and D_X optimization.(equation II)
        """
        D_X_loss = (D_X_real_loss + D_X_fake_loss) * 0.5
        """
        ∇_Θ just got computed by this one call!
        """
        D_X_loss.backward()
        """
        Now we just need to update all the parameters! 
        Θ_{k+1} = Θ_k − η * ∇_Θ ℓ(y_hat, y)
        """
        optimizer_D_X.step()

        D_X_running_loss+=D_X_loss.item()
        """
        *************************** Train Discriminator D_Y ***************************
        Discriminator: D_Y: G_XtoY(X) vs. Y.
        Now, real and fake loss of Discriminator D_Y.
        """
        """
        PyTorch stores gradients in a mutable data structure. So we need to set it to a clean state before we use it. 
        Otherwise, it will have old information from a previous iteration.
        """
        optimizer_D_Y.zero_grad()
        """
        Train D_Y with real real_Y images.
        5. Compute D_Y_real_loss, the real loss of discriminator D_Y on real real_Y images.
        """
        D_Y_real_out = D_Y(real_Y)
        D_Y_real_loss = mse_criterion(D_Y_real_out, torch.full(D_Y_real_out.size(), real_label, device=device))
        """
        Train with fake fake_Y images(History of generated images stored in the image buffer).
        6. Get generated fake_Y fake images from Image Buffer that look like domain Y and based on real images in domain X.
        """
        fake_Y = update_image_buffer_and_get_image(fake_Y_buffer,fake_Y,buffer_capacity)
        """
        7. Compute D_Y_fake_loss,the fake loss for discriminator D_Y on fake images.
        """
        D_Y_fake_out = D_Y(fake_Y)
        D_Y_fake_loss = mse_criterion(D_Y_fake_out, torch.full(D_Y_fake_out.size(), fake_label, device=device))
        
        """
        Back propagation
        As per the paper, I multiplied the loss for the discriminator by 0.5 during training, 
        in order to slow down updates to the discriminator relative to the generator model during training.
        8. Compute the total loss for D_Y, perform backpropagation and D_Y optimization.(Equation III)
        """
        D_Y_loss = (D_Y_real_loss + D_Y_fake_loss) * 0.5
        """
        ∇_Θ just got computed by this one call!
        """
        D_Y_loss.backward()
        """
        Now we just need to update all the parameters! 
        Θ_{k+1} = Θ_k − η * ∇_Θ ℓ(y_hat, y)
        """
        optimizer_D_Y.step()

        D_Y_running_loss+=D_Y_loss.item()
    
    """
    End training epoch.
    """
    end = time.time()
    total_train_time += (end-start)
    

    """
    Values for plot.
    """
    results["Epochs"].append(epoch)
    results["Total_time"].append(total_train_time)
    results["D_X_losses"].append(D_X_running_loss)
    results["D_Y_losses"].append(D_Y_running_loss)
    results["G_XtoY_losses"].append(G_XtoY_running_loss)
    results["G_YtoX_losses"].append(G_YtoX_running_loss)
    results["cycle_X_losses"].append(cycle_X_running_loss)
    results["cycle_Y_losses"].append(cycle_Y_running_loss)
    
    """
    Generating result for a specific train image of each domain to see the progress in fake image generation.
    """
    train_fake_Z, train_reconstructed_H = real_gen_recon_image(G_XtoY,G_YtoX,train_real_H)
    train_fake_H, train_reconstructed_Z = real_gen_recon_image(G_YtoX,G_XtoY,train_real_Z)


    generate_result([train_real_H, train_real_Z], 
                    [train_fake_Z, train_fake_H], 
                    [train_reconstructed_H, train_reconstructed_Z],
                    epoch,  
                    result_dir=cycleGAN_result_dir)
    """
    Generating result for a specific valiadtion image of each domain to see the progress in fake image generation.
    """
    if val_real_H is None or val_real_Z is None :
        pass
    else:
        G_XtoY = G_XtoY.eval()
        G_YtoX = G_YtoX.eval()
        
        val_fake_Z, val_reconstructed_H = real_gen_recon_image(G_XtoY,G_YtoX,val_real_H)
        val_fake_H, val_reconstructed_Z = real_gen_recon_image(G_YtoX,G_XtoY,val_real_Z)


        generate_result([val_real_H, val_real_Z], 
                        [val_fake_Z, val_fake_H], 
                        [val_reconstructed_H, val_reconstructed_Z],
                        epoch, 
                        result_dir=cycleGAN_validation_result_dir)
    
    """
    In PyTorch, the convention is to update the learning rate after every epoch.
    Updating learning rates.
    """
    lr_scheduler_G.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()
    
    """
    Showing lr deacy for few epochs.For 0 to 99 epoch lr is .0002.
    For the next
    Change in value for all optimizers' lr  are same hence showong only one lr.
    """
    if (epoch+1) in [99,100,120,180,199]:
        lr = optimizer_G.param_groups[0]['lr']
        print('optimizer_G\'s learning rate = %.7f' % lr,' at epoch : ', epoch)

    """
    Save the models checkpoint.
    """ 
    torch.save({'epoch'                   : epoch,
                'G_XtoY_state_dict'       : G_XtoY.state_dict(),
                'G_YtoX_state_dict'       : G_YtoX.state_dict(),
                'D_X_state_dict'          : D_X.state_dict(),
                'D_Y_state_dict'          : D_Y.state_dict(),
                'optimizer_G_state_dict'  : optimizer_G.state_dict(),
                'optimizer_D_X_state_dict': optimizer_D_X.state_dict(),
                'optimizer_D_Y_state_dict': optimizer_D_Y.state_dict(),
                'results'                 : results
                }, cycleGAN_checkpoint_dir + 'CycleGAN.pt')
    
"""
Creating DataFrame to hold losses which will be used to generate plot.
"""
results_df =  pd.DataFrame.from_dict(results)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




