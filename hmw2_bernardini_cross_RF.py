# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:35:25 2019

@author: bernardini carlo alberto 1225006
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import KFold
#%% DATA LOADING
import scipy.io as spio

mat = spio.loadmat('MNIST.mat', squeeze_me=True)
input_img_i = list(mat['input_images'])
input_img_i = input_img_i[:5000]
output_labels_i = list(mat['output_labels'].astype(int))
output_labels_i = output_labels_i[:5000]

num_img = len(input_img_i)
num_test = 1000
num_train = num_img - num_test

input_img = input_img_i.copy()
output_labels = output_labels_i.copy()

result = zip(input_img, output_labels)
resultList = list(result)
random.shuffle(resultList)
input_img_f, output_labels_f =  zip(*resultList)

#train set
x_train = list(input_img_f[num_test:])
y_train = list(output_labels_f[num_test:]) 

#test set
x_test = list(input_img_f[:num_test])
y_test = list(output_labels_f[:num_test])



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
#%% GRID SEARCH WITH CROSS VALIDATION - PARAM SETTING
kf = KFold(n_splits=3, random_state = 7, shuffle = True)
X = np.asarray(x_train)
Y = np.asarray(y_train)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
Nh1_grid = ["32", "64", "128"]
Nh2_grid = ["32", "64", "128"]
lr_grid = [ "1e-2" , "2e-2"]

loss_fn = nn.CrossEntropyLoss()


#%% GRID SEARCH K-FOLD
num_epochs = 10

loss_star = 100000000
Nh1_star = -1
Nh2_star = -1
lr_star = 0

for Nh1_string in Nh1_grid:
    for Nh2_string in Nh2_grid:
        for lr_string in lr_grid:
            Nh1 = int(Nh1_string)
            Nh2 = int(Nh2_string)
            lr = float(lr_string)
            fold = 0
            loss_avg_grid = []
            for train_index, test_index in kf.split(X):
                fold = fold + 1
                print('Fold', fold)
                net = Net(Nh1, Nh2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                net.to(device)
                optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
                X_train_kf, X_test_kf = X[train_index], X[test_index];
                Y_train_kf, Y_test_kf = Y[train_index], Y[test_index];
                for num_epoch in range(num_epochs):
                    print('Epoch', num_epoch + 1)
                    # Training
                    net.train() # Training mode (e.g. enable dropout)
                    # Eventually clear previous recorded gradients
                    optimizer.zero_grad()
                    conc_out = torch.Tensor().float().cuda()
                    conc_label = torch.Tensor().long().cuda()
                    for i in range(0, X_train_kf.shape[0]):
                        input_train = torch.tensor(X_train_kf[i]).float().view(-1, 784)
                        label_train = torch.tensor(Y_train_kf[i]).long().view(-1, 1).squeeze(1)
                        # Forward pass
                        out = net(input_train.cuda())
                        conc_out = torch.cat([conc_out, out])
                        conc_label = torch.cat([conc_label, label_train.cuda()])
                    # Evaluate loss
                    loss = loss_fn(conc_out.cuda(), conc_label.cuda())
                    # Backward pass
                    loss.backward()
                    # Update
                    optimizer.step()
                    # Print loss
                    print('\t Training loss ():', float(loss.data))
        
                    # Validation
                    net.eval() # Evaluation mode (e.g. disable dropout)
                    with torch.no_grad(): # No need to track the gradients
                        conc_out = torch.Tensor().float().cuda()
                        conc_label = torch.Tensor().long().cuda()
                        for i in range(0, X_test_kf.shape[0]):
                            # Get input and output arrays
                            input_test = torch.tensor(X_test_kf[i]).float().view(-1, 784)
                            label_test = torch.tensor(Y_test_kf[i]).long().view(-1, 1).squeeze(1)
                            # Forward pass
                            out = net(input_test.cuda())
                            # Concatenate with previous outputs
                            conc_out = torch.cat([conc_out, out])
                            conc_label = torch.cat([conc_label, label_test.cuda()])
                        # Evaluate global loss
                        test_loss = loss_fn(conc_out.cuda(), conc_label.cuda())
                        # Print loss
                        print('\t Validation loss:', float(test_loss.data))
                        if(num_epoch == num_epochs-1):
                            loss_avg_grid.append(float(test_loss.data))

            loss_avg_grid_scalar = np.mean(np.array(loss_avg_grid))
            if(loss_avg_grid_scalar < loss_star):
                loss_star = loss_avg_grid_scalar
                Nh1_star = Nh1
                Nh2_star = Nh2
                lr_star = lr
                print("tmp set found: Nh1-> " + str(Nh1_star) + " Nh2-> " + str(Nh2_star) + " lr-> " + str(lr_star))
print("final set found: Nh1-> " + str(Nh1_star) + " Nh2-> " + str(Nh2_star) + " lr-> " + str(lr_star))
print('optimal loss ' + str(loss_star))
#%% TRAINING OF BEST MODEL
input_img_i = list(mat['input_images'])
output_labels_i = list(mat['output_labels'].astype(int))

num_img = len(input_img_i)
num_test = 10000
num_train = num_img - num_test

input_img = input_img_i.copy()
output_labels = output_labels_i.copy()

result = zip(input_img, output_labels)
resultList = list(result)
random.shuffle(resultList)
input_img_f, output_labels_f =  zip(*resultList)

#train set
x_train = list(input_img_f[num_test:])
y_train = list(output_labels_f[num_test:]) 

#test set
x_test = list(input_img_f[:num_test])
y_test = list(output_labels_f[:num_test])

net_star = Net(1024, 1024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_star.to(device)
lr_star = 0.001
optimizer = optim.Adam(net_star.parameters(), lr=lr_star, weight_decay=5e-4)

conc_out = torch.Tensor().float()
conc_label = torch.Tensor().long()
num_epochs = 50
for num_epoch in range(num_epochs):
    
    print('Epoch', num_epoch + 1)
    # Training
    net_star.train() # Training mode (e.g. enable dropout)
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    conc_out = torch.Tensor().float().cuda()
    conc_label = torch.Tensor().long().cuda()
    for i in range(0, num_train):
        input_train = torch.tensor(x_train[i]).float().view(-1, 784)
        label_train = torch.tensor(y_train[i]).long().view(-1, 1).squeeze(1)
        # Forward pass
        out = net_star(input_train.cuda())
        conc_out = torch.cat([conc_out, out])
        conc_label = torch.cat([conc_label, label_train.cuda()])
    # Evaluate loss
    loss = loss_fn(conc_out.cuda(), conc_label.cuda())
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()
    # Print loss
    print('\t Training loss ():', float(loss.data))
    
#%% TEST PHASE
wrong_count = 0
for i in range(0, num_test):
    input_test = torch.tensor(x_test[i]).float().view(-1, 784)
    label_test = torch.tensor(y_test[i]).long().view(-1, 1).squeeze(1)
    out, res = net_star(input_test.cuda(), True)
    label_test = label_test.detach().numpy()
    if(res!=label_test):
        wrong_count = wrong_count + 1

accuracy = (num_test - wrong_count)/num_test
print('Accuracy: ', accuracy)

#%% FUCTION FOR VISUALIZE IMG
def view_overall(img, p_range):
    pixels_i = img.reshape((28, 28))
    pixels = np.transpose(pixels_i, (1, 0))
    pixels = np.flip(pixels, 0)
    p_range = p_range.detach().numpy()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(pixels)
    ax1.axis('off')
    ax1.invert_yaxis()
    ax2.barh(np.arange(10), p_range)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    
    plt.tight_layout()
    
    print("Most likely this is a : " + str(np.argmax(p_range)))

#%% TEST RANDOM SAMPLE
tryy = random.randint(0, num_test)
sample = x_test[tryy]
input_test = torch.tensor(sample).float().view(-1, 784)
label_test = torch.tensor(y_test[tryy]).long().view(-1, 1).squeeze(1)
# Forward pass
out, belong_class = net_star(input_test.cuda(), True)
out_cpu = out.cpu()
softmaxi = nn.functional.softmax(out_cpu, dim=1).squeeze()
view_overall(sample, softmaxi)
print("Net return class: ", belong_class)

#%% SAVING THE NET STATE
#net_state_dict = net_star.state_dict()
#torch.save(net_state_dict, 'net_parameters_reupload.torch')

#%% RECEPTIVE FIELDS SUBSET HIDDEN NEURONS
W1 = net_star.fc1.weight.cpu().detach().numpy()
RF_l1 = []
for k in range(4):
    RF_l1.append(np.reshape(W1[k+10],(28,28)))
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(10,3), ncols=4)
ax1.imshow(RF_l1[0])
ax2.imshow(RF_l1[1])
ax3.imshow(RF_l1[2])
ax4.imshow(RF_l1[3])
fig.suptitle('RF of first hidden layer neurons')

W2_i = net_star.fc2.weight.cpu().detach().numpy()
W2 = np.matmul(W2_i,W1)
RF_l2 = []
for k in range(4):
    RF_l2.append(np.reshape(W2[k+10],(28,28)))
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(10,3), ncols=4)
ax1.imshow(RF_l2[0])
ax2.imshow(RF_l2[1])
ax3.imshow(RF_l2[2])
ax4.imshow(RF_l2[3])
fig.suptitle('RF of second hidden layer neurons')

W3_i = net_star.fc3.weight.cpu().detach().numpy()
W3 = np.matmul(W3_i,W2)
RF_l3 = []
for k in range(4):
    RF_l3.append(np.reshape(W3[k+4],(28,28)))
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(10,3), ncols=4)
ax1.imshow(RF_l3[0])
ax2.imshow(RF_l3[1])
ax3.imshow(RF_l3[2])
ax4.imshow(RF_l3[3])
fig.suptitle('RF of output layer neurons')







