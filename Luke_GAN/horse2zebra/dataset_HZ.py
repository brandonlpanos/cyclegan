# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:24:47 2023

@author: lukem
"""

import os
from torch.utils.data import Dataset
from PIL import Image



class ImageDataset(Dataset):
    def __init__(self, image_dir, is_train, image_type,transform):
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
        print(image.shape)
        return image
 
    def __len__(self):
        return len(self.image_filename_lst)