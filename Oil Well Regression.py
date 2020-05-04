# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:05:41 2019

@author: abhishek
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:07:46 2019

@author: abhishek
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 00:23:04 2019

@author: abhishek
"""

#!/usr/bin/env python
# coding: utf-8

# To Do
# 1. use plotly or bokeh for interactive graphs 
# 2. Importance of correlations and use in heat maps / cluster maps
# 
# Important
# 1. Treat the datasets as individuals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Reading all the training datasets 
ow1_train = pd.read_excel("Oil_Reservoir_Data_1_train.xlsx")
ow2_train = pd.read_excel("Oil_Reservoir_Data_2_train.xlsx")
ow3_train = pd.read_excel("Oil_Reservoir_Data_3_train.xlsx")

# Renaming the Last column as FLAG
ow1_train = ow1_train.rename(columns={ ow1_train.columns[-1]: "FLAG" })
ow2_train = ow2_train.rename(columns={ ow2_train.columns[-1]: "FLAG" })
ow3_train = ow3_train.rename(columns={ ow3_train.columns[-1]: "FLAG" })

# Converting Grac to 0, OK to 1 and Good to 2
ow1_train["FLAG"]= ow1_train["FLAG"].apply(lambda x: 0 if x == "Frac" else ( 1 if x == "OK" else 2))
ow2_train["FLAG"]= ow2_train["FLAG"].apply(lambda x: 0 if x == "Frac" else ( 1 if x == "OK" else 2))
ow3_train["FLAG"]= ow3_train["FLAG"].apply(lambda x: 0 if x == "Frac" else ( 1 if x == "OK" else 2))

# Reading all the testing datasets 
ow1_test = pd.read_excel("Oil_Reservoir_Data_1_test.xlsx")
ow2_test = pd.read_excel("Oil_Reservoir_Data_2_test.xlsx")
ow3_test = pd.read_excel("Oil_Reservoir_Data_3_test.xlsx")

# Renaming the Last column as FLAG
ow1_test = ow1_test.rename(columns={ ow1_test.columns[-1]: "FLAG" })
ow2_test = ow2_test.rename(columns={ ow2_test.columns[-1]: "FLAG" })
ow3_test = ow3_test.rename(columns={ ow3_test.columns[-1]: "FLAG" })

# Converting Grac to 0, OK to 1 and Good to 2
ow1_test["FLAG"]= ow1_test["FLAG"].apply(lambda x: 0 if x == "Frac" else ( 1 if x == "OK" else 2))
ow2_test["FLAG"]= ow2_test["FLAG"].apply(lambda x: 0 if x == "Frac" else ( 1 if x == "OK" else 2))
ow3_test["FLAG"]= ow3_test["FLAG"].apply(lambda x: 0 if x == "Frac" else ( 1 if x == "OK" else 2))

# Graph showing GR vs Frequency from Oil Well 1
plt.figure(figsize=(18,6))
plt.hist(ow1_train.GR, bins = 20)
plt.show()

# Graph showing the correlation matrix heatmap
correlations = ow1_train.corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlations,cbar=True, annot=True)
plt.show()

# Dropping the columns to be predicted and creating training dataset
X_train1 = ow1_train.drop(columns=['logK','FLAG','Phi'])
X_test1 = ow1_test.drop(columns=['logK','FLAG','Phi'])
X_train2 = ow2_train.drop(columns=['logK','FLAG','Phi'])
X_test2 = ow2_test.drop(columns=['logK','FLAG','Phi'])
X_train3 = ow3_train.drop(columns=['logK','FLAG','Phi'])
X_test3 = ow3_test.drop(columns=['logK','FLAG','Phi'])

# selecting the logK column to make the test dataset
Y_train1_logk = pd.DataFrame(ow1_train.logK)
Y_test1_logk = pd.DataFrame(ow1_test.logK)
Y_train2_logk= pd.DataFrame(ow2_train.logK)
Y_test2_logk= pd.DataFrame(ow2_test.logK)
Y_train3_logk = pd.DataFrame(ow3_train.logK)
Y_test3_logk = pd.DataFrame(ow3_test.logK)

# selecting the Phi column to make the test dataset
Y_train1_phi = pd.DataFrame(ow1_train.Phi)
Y_test1_phi = pd.DataFrame(ow1_test.Phi)
Y_train2_phi = pd.DataFrame(ow2_train.Phi)
Y_test2_phi = pd.DataFrame(ow2_test.Phi)
Y_train3_phi = pd.DataFrame(ow3_train.Phi)
Y_test3_phi = pd.DataFrame(ow3_test.Phi)

## Testing for LogK in Dataset 1

# Creating training tensor
X = torch.Tensor(X_train1.as_matrix()).float()
Y = torch.Tensor(Y_train1_logk.as_matrix()).float()

#creating testing tensor
X_test = torch.Tensor(X_test1.as_matrix()).float()
Y_test = torch.Tensor(Y_test1_logk.as_matrix()).float()

# Setting parameters for Neural Network
input_size = 8
num_classes = 1
hidden = 5
learning_rate = 0.01
num_epochs = 500

# Defining neural network structure
class Regression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Regression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = self.linear2(out)
        return out


# defining neural network
reg_model = Regression(input_size, num_classes)

# defining loss function
criterion = nn.MSELoss()
criterionBimodal = nn.MSELoss(reduction='none')

# defining optimiser function
optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    
    # Performing forward pass
    Y_pred = reg_model(X)
    Y_pred = Y_pred.view(len(Y_pred))
    
    # Calculating loss 
    loss = criterion(Y_pred, Y)
        
    all_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print('Training Epoch: [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    optimizer.zero_grad()

    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimizer.step()

# Showing Graph of al losses
plt.figure()
plt.plot(all_losses)
plt.show()

# Get predictions for test data
Y_pred_test = reg_model(X_test)
test_loss = criterion(Y_pred_test.view(Y_pred_test.size(0)), Y_test)

# Print loss on the test data
print('test loss: %f' % test_loss.item())


 
## Testing for LogK in Dataset 2
 
X = torch.Tensor(X_train2.as_matrix()).float()
Y = torch.Tensor(Y_train2_logk.as_matrix()).float()
 
X_test = torch.Tensor(X_test2.as_matrix()).float()
Y_test = torch.Tensor(Y_test2_logk.as_matrix()).float()
 
 # defining neural network
reg_model = Regression(input_size, num_classes)

# defining loss function
criterion = nn.MSELoss()
criterionBimodal = nn.MSELoss(reduction='none')

# defining optimiser function
optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    
    # Performing forward pass
    Y_pred = reg_model(X)
    Y_pred = Y_pred.view(len(Y_pred))
    
    # Calculating loss 
    loss = criterion(Y_pred, Y)
    all_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print('Training Epoch: [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    optimizer.zero_grad()

    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimizer.step()

# Showing Graph of al losses
plt.figure()
plt.plot(all_losses)
plt.show()

# Get predictions for test data
Y_pred_test = reg_model(X_test)
test_loss = criterion(Y_pred_test.view(Y_pred_test.size(0)), Y_test)

# Print loss on the test data
print('test loss: %f' % test_loss.item())

 
 
 ## Testing for LogK in Dataset 3
 
X = torch.Tensor(X_train3.as_matrix()).float()
Y = torch.Tensor(Y_train3_logk.as_matrix()).float()
 
X_test = torch.Tensor(X_test3.as_matrix()).float()
Y_test = torch.Tensor(Y_test3_logk.as_matrix()).float()
 
 # defining neural network
reg_model = Regression(input_size, num_classes)

# defining loss function
criterion = nn.MSELoss()
criterionBimodal = nn.MSELoss(reduction='none')

# defining optimiser function
optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    
    # Performing forward pass
    Y_pred = reg_model(X)
    Y_pred = Y_pred.view(len(Y_pred))
    
    # Calculating loss 
    loss = criterion(Y_pred, Y)
        
    all_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print('Training Epoch: [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    optimizer.zero_grad()

    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimizer.step()

# Showing Graph of al losses
plt.figure()
plt.plot(all_losses)
plt.show()

# Get predictions for test data
Y_pred_test = reg_model(X_test)
test_loss = criterion(Y_pred_test.view(Y_pred_test.size(0)), Y_test)

# Print loss on the test data
print('test loss: %f' % test_loss.item())

 
 ## Testing for Phi in Dataset 1
 
X = torch.Tensor(X_train1.as_matrix()).float()
Y = torch.Tensor(Y_train1_phi.as_matrix()).float()
 
X_test = torch.Tensor(X_test1.as_matrix()).float()
Y_test = torch.Tensor(Y_test1_phi.as_matrix()).float()
 # defining neural network
reg_model = Regression(input_size, num_classes)

# defining loss function
criterion = nn.MSELoss()
criterionBimodal = nn.MSELoss(reduction='none')

# defining optimiser function
optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    
    # Performing forward pass
    Y_pred = reg_model(X)
    Y_pred = Y_pred.view(len(Y_pred))
    
    # Calculating loss 
    loss = criterion(Y_pred, Y)
        
    all_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print('Training Epoch: [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    optimizer.zero_grad()

    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimizer.step()

# Showing Graph of al losses
plt.figure()
plt.plot(all_losses)
plt.show()

# Get predictions for test data
Y_pred_test = reg_model(X_test)
test_loss = criterion(Y_pred_test.view(Y_pred_test.size(0)), Y_test)

# Print loss on the test data
print('test loss: %f' % test_loss.item())

## Testing for Phi in Dataset 2

X = torch.Tensor(X_train2.as_matrix()).float()
Y = torch.Tensor(Y_train2_phi.as_matrix()).float()

X_test = torch.Tensor(X_test2.as_matrix()).float()
Y_test = torch.Tensor(Y_test2_phi.as_matrix()).float()

# defining neural network
reg_model = Regression(input_size, num_classes)

# defining loss function
criterion = nn.MSELoss()
criterionBimodal = nn.MSELoss(reduction='none')

# defining optimiser function
optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    
    # Performing forward pass
    Y_pred = reg_model(X)
    Y_pred = Y_pred.view(len(Y_pred))
    
    # Calculating loss 
    loss = criterion(Y_pred, Y)
        
    all_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print('Training Epoch: [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    optimizer.zero_grad()

    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimizer.step()

# Showing Graph of al losses
plt.figure()
plt.plot(all_losses)
plt.show()

# Get predictions for test data
Y_pred_test = reg_model(X_test)
test_loss = criterion(Y_pred_test.view(Y_pred_test.size(0)), Y_test)

# Print loss on the test data
print('test loss: %f' % test_loss.item())

## Testing for Phi in Dataset 3

X = torch.Tensor(X_train3.as_matrix()).float()
Y = torch.Tensor(Y_train3_phi.as_matrix()).float()

X_test = torch.Tensor(X_test3.as_matrix()).float()
Y_test = torch.Tensor(Y_test3_phi.as_matrix()).float()
# defining neural network
reg_model = Regression(input_size, num_classes)

# defining loss function
criterion = nn.MSELoss()
criterionBimodal = nn.MSELoss(reduction='none')

# defining optimiser function
optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    
    # Performing forward pass
    Y_pred = reg_model(X)
    Y_pred = Y_pred.view(len(Y_pred))
    
    # Calculating loss 
    loss = criterion(Y_pred, Y)
        
    all_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print('Training Epoch: [%d/%d], Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    optimizer.zero_grad()

    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimizer.step()

# Showing Graph of al losses
plt.figure()
plt.plot(all_losses)
plt.show()

# Get predictions for test data
Y_pred_test = reg_model(X_test)
test_loss = criterion(Y_pred_test.view(Y_pred_test.size(0)), Y_test)

# Print loss on the test data
print('test loss: %f' % test_loss.item())
