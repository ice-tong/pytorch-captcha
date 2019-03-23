# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:07:17 2019

@author: icetong
"""

import torch
import torch.nn as nn
from models import CNN
from datasets import CaptchaData
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plot

model_path = './checkpoints/model.pth'

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
alphabet = ''.join(source)

def predict(img_dir='./data/test'):
    transforms = Compose([ToTensor()])
    dataset = CaptchaData(img_dir, transform=transforms)
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))
    
    for k, (img, target) in enumerate(dataset):
        img = img.view(1, 3, 100, 180).cuda()
        target = target.view(1, 4*36).cuda()
        output = cnn(img)
        
        output = output.view(-1, 36)
        target = target.view(-1, 36)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output = output.view(-1, 4)[0]
        target = target.view(-1, 4)[0]
        
        print('pred: '+''.join([alphabet[i] for i in output.cpu().numpy()]))
        print('true: '+ ''.join([alphabet[i] for i in target.cpu().numpy()]))
        
        plot.imshow(img.permute((0, 2, 3, 1))[0].cpu().numpy())
        plot.show()
        
        if k >= 10: break
        
if __name__=="__main__":
    predict()