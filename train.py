# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:20:40 2019

@author: icetong
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from models import CNN
from datasets import CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

import time
import os

batch_size = 128
base_lr = 0.001
max_epoch = 200
model_path = './checkpoints/model.pth'
restor = False

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

def calculat_acc(output, target):
    output, target = output.view(-1, 36), target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc

def train():
    transforms = Compose([ToTensor()])
    train_dataset = CaptchaData('./data/train', transform=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, 
                             shuffle=True, drop_last=True)
    test_data = CaptchaData('./data/test', transform=transforms)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, 
                                  num_workers=0, shuffle=True, drop_last=True)
    cnn = CNN()
    if torch.cuda.is_available():
        cnn.cuda()
    if restor:
        cnn.load_state_dict(torch.load(model_path))
#        freezing_layers = list(cnn.named_parameters())[:10]
#        for param in freezing_layers:
#            param[1].requires_grad = False
#            print('freezing layer:', param[0])
    
    optimizer = torch.optim.Adam(cnn.parameters(), lr=base_lr)
    criterion = nn.MultiLabelSoftMarginLoss()
    
    for epoch in range(max_epoch):
        start_ = time.time()
        
        loss_history = []
        acc_history = []
        cnn.train()
        for img, target in train_data_loader:
            img = Variable(img)
            target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = calculat_acc(output, target)
            acc_history.append(acc)
            loss_history.append(loss)
        print('train_loss: {:.4}|train_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))
        
        loss_history = []
        acc_history = []
        cnn.eval()
        for img, target in test_data_loader:
            img = Variable(img)
            target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            
            acc = calculat_acc(output, target)
            acc_history.append(acc)
            loss_history.append(float(loss))
        print('test_loss: {:.4}|test_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time()-start_))
        torch.save(cnn.state_dict(), model_path)

if __name__=="__main__":
    train()
    pass