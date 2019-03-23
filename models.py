# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:17:51 2019

@author: icetong
"""

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_class=36, num_char=4):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
                #batch*3*180*100
                nn.Conv2d(3, 16, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                #batch*16*90*50
                nn.Conv2d(16, 64, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                #batch*64*45*25
                nn.Conv2d(64, 512, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #batch*512*22*12
                nn.Conv2d(512, 512, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #batch*512*11*6
                )
        self.fc = nn.Linear(512*11*6, self.num_class*self.num_char)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512*11*6)
        x = self.fc(x)
        return x