# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:44:45 2019

@author: berna
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.io as spio

#%% Neural Network

### Define the network class
class Net(nn.Module):
    
    def __init__(self, Nh1, Nh2):
        super().__init__()
        
        self.fc1 = nn.Linear(784,Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, 10)
        
        self.act = nn.ReLU()
        self.act_last = nn.Softmax(dim=1)
        
    def forward(self, x, additional_out=False):
        
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.fc3(x)
        
        if additional_out:
            return out, np.argmax(self.act_last(self.fc3(x)).cpu().detach().numpy())
        
        return out
#%% DATA LOADING
mat = spio.loadmat('MNIST.mat', squeeze_me=True)
input_img = list(mat['input_images'])
output_labels = list(mat['output_labels'].astype(int))

#%% NET UPLOAD FROM THE OTHER .PY FILE
net = Net(128,64)

net_state_dict = torch.load('net_parameters_reupload.torch')
net.load_state_dict(net_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

#%% TEST PHASE
num_samples = len(input_img)
wrong_count = 0
for i in range(0, num_samples):
    input_test = torch.tensor(input_img[i]).float().view(-1, 784)
    label_test = torch.tensor(output_labels[i]).long().view(-1, 1).squeeze(1)
    out, res= net(input_test.cuda(), True)
    label_test.detach().numpy()
    if(res!=label_test):
        wrong_count = wrong_count + 1

accuracy = (num_samples - wrong_count)/num_samples
print('Accuracy: ', accuracy)
