import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import sunkit_image.enhance as enhance

def create_loaders(data_directory,test_percent, batch_size, sdo_channels, iris_channel):
    """
    Creates data loaders for training and testing based on the provided parameters.
    Args:
        test_percent (float, optional): The percentage of data to be used for testing.
            Defaults to 0.8 (80% for training and 20% for testing).
        batch_size (int, optional): The batch size for the data loaders.
            Defaults to 64.
    Returns:
        tuple: A tuple containing the training and testing data loaders.
    Raises:
        AssertionError: If the observation directories for SDO and IRIS data do not match.
    """

    assert set(os.listdir(data_directory +"sdo/")) == set(os.listdir(data_directory +"iris/")), 'obs not in both directories'

    # Get the list of observations for SDO data
    obs_list = os.listdir(data_directory +"sdo/")

    # Split the observations into training and testing sets
    train_set, test_set = train_test_split(obs_list, test_size=test_percent, random_state=42)
    train_obs_list = train_set
    test_obs_list = test_set

    # Create the training and testing datasets
    train_dataset = AIAIRISDataset(data_directory,
        sdo_channels=sdo_channels, iris_channel=iris_channel, obs_list=train_obs_list)
    test_dataset = AIAIRISDataset(data_directory,
        sdo_channels=sdo_channels, iris_channel=iris_channel, obs_list=test_obs_list)

    # Create the training and testing data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class AIAIRISDataset(Dataset):
    """
    Custom dataset class for AIA and IRIS images.
    Args:
        sdo_channels (list, optional): List of SDO channels to include. Defaults to ['304'].
        iris_channel (str, optional): IRIS channel to include. Defaults to '1400'.
        obs_list (list, optional): List of observations to include. Defaults to None.
    Attributes:
        iris_paths (ndarray): Array of paths to IRIS images.
        sdo_paths (ndarray): Array of paths to SDO images.
        transform (Compose): Composition of image transformations.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the preprocessed images at the given index.
    """

    def __init__(self, data_directory,sdo_channels=['304'], iris_channel='1400', obs_list=None):
        super(AIAIRISDataset, self).__init__()

        # Collect paths to IRIS images
        iris_paths = []
        for obs in obs_list:
            path_to_iris_obs = data_directory +f'iris/{obs}/{iris_channel}/'
            for file in os.listdir(path_to_iris_obs):
                iris_paths.append(f'{path_to_iris_obs}/{file}')

        # Collect paths to SDO images
        sdo_paths = []
        for obs in obs_list:
            path_to_sdo_obs = data_directory +f'sdo/{obs}/'
            for sdo_channel in sdo_channels:
                path_to_sdo_obs_channel = f'{path_to_sdo_obs}/{sdo_channel}/'
                for file in os.listdir(path_to_sdo_obs_channel):
                    sdo_paths.append(f'{path_to_sdo_obs_channel}/{file}')

        # Randomly shuffle the indices for both IRIS and SDO images. Note that results could imporve if images alligned in time
        iris_rand_ints = np.random.choice(
            len(iris_paths), size=len(iris_paths), replace=False)
        sdo_rand_ints = np.random.choice(
            len(sdo_paths), size=len(sdo_paths), replace=False)

        # Store the shuffled paths
        self.iris_paths = np.array(iris_paths)[iris_rand_ints]
        self.sdo_paths = np.array(sdo_paths)[sdo_rand_ints]





        # Define the image transformations
        # Note, we have to pad the images to the same size, have found max im size to be (1096, 1017) and hard coded. Mean is (463, 441)
    
        

    def __len__(self):
        return min(len(self.iris_paths), len(self.sdo_paths))
    
    
    def removeout(self,arr):
        mean = np.median(arr)

        standard_deviation = np.std(arr)
        
        #distance_from_mean = abs(arr - mean)
        
        max_deviations = 0.01
        
        not_outlier = max_deviations * standard_deviation
        
        no_outliers = np.clip(arr,mean-not_outlier,mean+not_outlier)
        
        
        #print(mean, mean-not_outlier, mean+not_outlier)
        
        return no_outliers

    def __getitem__(self, idx):

        iris_im = np.load(self.iris_paths[idx])
        sdo_im = np.load(self.sdo_paths[idx])
        
        iris_mean = np.average(iris_im)
        aia_mean = np.average(sdo_im)

        iris_std = np.std(iris_im)
        aia_std = np.std(sdo_im)
        
        #transformer = RobustScaler().fit(sdo_im)

        #sdo_im = transformer.transform(sdo_im)
        
        out_sdo_im = self.removeout(sdo_im)
        out_iris_im = self.removeout(iris_im)
        
        
        
        norm_transform = transforms.Compose([
            #NormalizeMinMax_sklearn,
            NormalizeMinMax(),
            #transforms.CenterCrop([463, 463]),
            #transforms.RandomRotation(180),
            transforms.ToTensor(),
            #transforms.Normalize(mean=sdo_im_mean, std=sdo_im_std)

        ])
        
        tens_transform = transforms.Compose([
            transforms.ToTensor(),

        ])
        
        

        out_iris_im = Image.fromarray(out_iris_im)
        out_sdo_im = Image.fromarray(out_sdo_im)
        
        iris_im = Image.fromarray(iris_im)
        sdo_im = Image.fromarray(sdo_im)
        
        augmented_iris_im = norm_transform(out_iris_im)
        augmented_sdo_im = norm_transform(out_sdo_im)
        
        tens_iris_im = tens_transform(iris_im)
        tens_sdo_im = tens_transform(sdo_im)
        
        #print(torch.max(augmented_sdo_im), torch.min(augmented_sdo_im))
        #augmented_sdo_im = torch.clamp(augmented_sdo_im, min = -1, max = 1)

        return{'A': augmented_iris_im, 'B':augmented_sdo_im, 'C':tens_iris_im, 'D':tens_sdo_im}
    


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
    
"""class NormalizeMinMax_iris(torch.nn.Module):
    
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

        min_value = -2.5
        max_value = 25
        
        #image = torch.clamp(image, max = max_value, min = min_value)
        
        print(min_value)
        print(max_value)

        # Normalize image to range [-1, 1]
        normalized_image = (image - min_value) / (max_value - min_value)
        normalized_image = (normalized_image * 2) - 1
        
        normalized_image = torch.clamp(normalized_image, max = 1, min = -1)
        
        

        if isinstance(img, Image.Image):
            # Convert tensor back to PIL image
            normalized_image = transforms.functional.to_pil_image(normalized_image)

        return normalized_image"""





