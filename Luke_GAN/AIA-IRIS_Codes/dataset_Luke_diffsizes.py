import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import cv2 as cv

def create_loaders():
    train_dataset = AIAIRISDataset('train')
    test_dataset = AIAIRISDataset('test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader


class AIAIRISDataset(Dataset):
    def __init__(self, image_dir, is_train, image_type):
        super(AIAIRISDataset, self).__init__()
        
        self.train_or_test='train' if is_train else 'test'
        self.image_dir = './' + image_dir
        self.image_type=image_type
        self.image_path = os.path.join(self.image_dir,'{}'.format(self.image_type)+'_'+self.train_or_test)
        self.image_filename_lst = [x for x in sorted(os.listdir(self.image_path))]

        
    def __len__(self):
        return len(self.image_filename_lst)

    def removeout(self,arr,max_deviations):
        
        #print("Removal Shape: ",arr.shape)
        
        """if len(arr.shape) > 1:
            
            for i in arr:
                mean = np.median(i)
                standard_deviation = np.std(i)             
                not_outlier = max_deviations * standard_deviation 
                i = np.clip(i,mean-not_outlier,mean+not_outlier)
                
        else:
            print("Here")"""
            
        mean = np.median(arr)
        standard_deviation = np.std(arr)
        not_outlier = max_deviations * standard_deviation
        aug_arr = np.clip(arr,mean-not_outlier,mean+not_outlier)
        
        
        return aug_arr

    def __getitem__(self, idx):
        
        
        
        image_file = os.path.join(self.image_path, self.image_filename_lst[idx])
        #out_im = np.load(image_file).astype(np.uint16)
        
        

        
        
        #out_im = np.float32(out_im)
        
        #print(image.shape)
        
        
        #grayimg = cv.cvtColor(out_im, cv.COLOR_BGR2GRAY)
            
        
        norm_iris_transform = transforms.Compose([
            transforms.Resize(size=286),
            transforms.CenterCrop(256),
            NormalizeMinMax(),
            transforms.RandomHorizontalFlip(0.4),
            transforms.RandomVerticalFlip(0.4),
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.5), std=(0.5)),

        ])
        
        
        
        norm_sdo_transform = transforms.Compose([
            transforms.Resize(size=286),
            transforms.CenterCrop(256),
            NormalizeMinMax(),
            transforms.RandomHorizontalFlip(0.4),
            transforms.RandomVerticalFlip(0.4),
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.5), std=(0.5)),

        ])
        
        
        if self.image_type == 'iris':
            #out_im = self.removeout(out_im,1)
            out_im = np.load(image_file).astype(np.uint8)
            
            """hist,bins = np.histogram(out_im.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(out_im.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('cdf','histogram iris'), loc = 'upper left')
            plt.show()"""
            
            
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            equ = clahe.apply(out_im)
            
            #equ = cv.equalizeHist(out_im)
            
            
            """hist,bins = np.histogram(equ.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(equ.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('cdf','histogram iris'), loc = 'upper left')
            plt.show()"""
            
            #equ = equ.astype(np.float32)
            
            out_im = Image.fromarray(equ)
            augmented_im = norm_iris_transform(out_im)

            
        elif self.image_type == 'aia':
            
            out_im = np.load(image_file)#.astype(np.uint16)
            
            out_im = ((out_im - out_im.min()) * (1/(out_im.max() - out_im.min()) * 255)).astype('uint8')
            
            #out_im = self.removeout(out_im,100)
            """hist,bins = np.histogram(out_im.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(out_im.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('cdf','histogram aia'), loc = 'upper left')
            plt.show()"""
            
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            equ = clahe.apply(out_im)
            
            #equ = cv.equalizeHist(out_im)
            
            """hist,bins = np.histogram(equ.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(equ.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('cdf','histogram aia'), loc = 'upper left')
            plt.show()"""
            
            #equ = equ.astype(np.float16)
            
            out_im = Image.fromarray(equ)
            augmented_im = norm_sdo_transform(out_im)
            
            
        
        #print("out size: ",out_im.shape)


        #print(augmented_im.shape)
        
        return augmented_im
        
    


# might want to consider using transformations from skimage import exposure, filters, util
class NormalizeMinMax(torch.nn.Module):
    """
    A module to normalize images for GAN training by performing min-max normalization.
    This module takes either a PIL image or a torch tensor as input and normalizes it
    to the range [-1, 1] using min-max normalization. It is particularly useful for
    preparing images for training Generative Adversarial Networks (GANs).
    Args:
        None
    Methods:
        forward(img):
            Normalize the input image tensor to the range [-1, 1].
    Examples:
        # Create an instance of NormalizeMinMax
        normalizer = NormalizeMinMax()
        # Normalize a PIL image
        pil_image = Image.open('image.jpg')
        normalized_image = normalizer(pil_image)
        # Normalize a torch tensor
        tensor_image = torch.randn(3, 256, 256)
        normalized_image = normalizer(tensor_image)
    """
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if isinstance(img, Image.Image):
            # Convert PIL image to tensor
            image = transforms.functional.to_tensor(img)
        elif isinstance(img, torch.Tensor):
            image = img
        else:
            raise TypeError(f"Unexpected type {type(img)}")

        min_value = torch.min(image)
        max_value = torch.max(image)
        
        #print(min_value)
        #print(max_value)
        

        # Normalize image to range [-1, 1]
        normalized_image = (image - min_value) / (max_value - min_value)
        normalized_image = (normalized_image * 2) - 1
        
        
        
        
        #normalized_image = torch.clamp(normalized_image, max =1,min = -1)
        
        

        if isinstance(img, Image.Image):
            # Convert tensor back to PIL image
            normalized_image = transforms.functional.to_pil_image(normalized_image)

        return normalized_image


        
