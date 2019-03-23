# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:49:21 2019

@author: icetong
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
alphabet = ''.join(source)

def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert('RGB')

def make_dataset(data_path, alphabet, num_class, num_char):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        target_str = img_name.split('.')[0]
        assert len(target_str) == num_char
        target = []
        for char in target_str:
            vec = [0] * num_class
            vec[alphabet.find(char)] = 1
            target += vec
        samples.append((img_path, target))
    return samples

class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=36, num_char=4, 
                 transform=None, target_transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = make_dataset(self.data_path, self.alphabet, 
                                    self.num_class, self.num_char)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)