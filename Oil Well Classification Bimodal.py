# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:51:32 2019

@author: abhishek
"""


import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import time
import numpy as np

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
plt.hist(ow1_train.GR, bins = 20, histtype='bar' , ec='black')
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

# Selecting the last column for Y
Y_train1_class =  ow1_train.iloc[:, 10]
Y_test1_class = ow1_test.iloc[:, 10]
Y_train2_class =  ow2_train.iloc[:, 10]
Y_test2_class = ow2_test.iloc[:, 10]
Y_train3_class =  ow3_train.iloc[:, 10]
Y_test3_class = ow3_test.iloc[:, 10]

# Setting parameters for Neural Network
input_neurons = 8
hidden_neurons = 10
output_neurons = 3
learning_rate = 0.01
num_epochs = 500

# Defining neural network structure
class Classification(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Classification, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h_input = self.hidden(x)
        h_output = F.relu(h_input)
        y_pred = self.out(h_output)
        return y_pred

# Creating training and testing tensor
X = torch.Tensor(X_train1.values).float()
Y = torch.Tensor(Y_train1_class.values).long()
X_test = torch.Tensor(X_test1.values).float()
Y_test = torch.Tensor(Y_test1_class.values).long()

# defining neural network
net = Classification(input_neurons, hidden_neurons, output_neurons)

# defining loss function
loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

criterionBimodal = nn.CrossEntropyLoss(reduction='none')
# defining optimiser function
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    # Performing forward pass
    Y_pred = net(X)
    
    # Calculating loss 
    loss = loss_func(Y_pred, Y)
    
    # for calculating loss at epoch 10
    if(epoch==10):
        lossbdr = criterionBimodal(Y_pred, Y)
        lossat0 = lossbdr.detach().numpy()
        
    # for calculating loss at epoch 50
    if(epoch==50):
        lossbdr100 = criterionBimodal(Y_pred, Y)
        lossat100 = lossbdr100.detach().numpy()
        
    if((epoch+1)%50 == 0):
        lossbdr = criterionBimodal(Y_pred, Y)
        # Checking if pattern loss > mean loss
        indexes = lossbdr>loss.mean()
        subset = ((indexes==1).nonzero()).squeeze()
        inversesubset = ((indexes==0).nonzero()).squeeze()
        
        # Creating new tensors and removing the patterns with loss > mean loss in X and Y
        X_bdr = X[subset.data]
        X = X[inversesubset.data]
        Y_bdr = Y[subset.data]
        Y = Y[inversesubset.data]
        
        #Creating tensor for loss in the subse
        error_subset = lossbdr[subset.data]
        X_bdr_mean = error_subset
        
        # Calculating variance
        variance = np.var(X_bdr_mean.data.numpy())
        
        #Stopping the training if variance <0.001 and putting the remaining subset data to X
        if(variance<0.001):
            X = torch.cat((X, X_bdr), dim=0)
            Y = torch.cat((Y, Y_bdr), dim=0)
            break
        #Calculating standard deviation
        X_bdr_std = error_subset.std()
        
        # getting the indexes where error > mean + alpha*standard deviation
        X_bdr_sum = X_bdr_mean+(X_bdr_std*0.3)
        final_index = X_bdr_mean.mean()>=X_bdr_sum
        
        #creatin final subset
        final_subset = ((final_index==0).nonzero()).squeeze();
        X_final = X_bdr[final_subset.data]
        Y_final = Y_bdr[final_subset.data]
        
        # adding the remaining data back to X
        if(X_final.dim()!=1 and Y_final.dim()!=1 and X_final.dim()==X.dim() and Y_final.dim()==Y.dim()):
            X = torch.cat((X, X_final), dim=0)
            Y = torch.cat((Y, Y_final), dim=0)
        
        removed = ((final_index==1).nonzero()).squeeze();
        count = list(removed.size())
        
        print("Removed Outliers ", count)
    
    endbdr = time.time()
    all_losses.append(loss.item())
    if ((epoch+1) % 10 == 0):
        _, predicted = torch.max(Y_pred, 1)
        print('Epoch [%d/%d] Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    net.zero_grad()
    
    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimiser.step()

# Showing pattern error graph at epoch 10 
plt.figure()
plt.hist(lossat0,  histtype='bar' , ec='black')
plt.show()

# Showing pattern error graph at epoch 50 
plt.figure()
plt.hist(lossat100,  histtype='bar' , ec='black')
plt.show()

# Showing Graph of all losses
plt.figure()
plt.plot(all_losses)
plt.show()

#Creating confusion matrix
confusion = torch.zeros(output_neurons, output_neurons)

# Getting training accuracy
Y_pred = net(X)
_, predicted = torch.max(Y_pred, 1)
total = predicted.size(0)
correct = sum(predicted.data.numpy() == Y.data.numpy())
print('Training Accuracy: %.2f %%' % (100 * correct / total))

for i in range(X.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

# getting predictions for X test
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
confusion_test = torch.zeros(output_neurons, output_neurons)
for i in range(X_test1.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for testing:')
print(confusion_test)


# predicting FLAG for Oil Well 2
X = torch.Tensor(X_train2.values).float()
Y = torch.Tensor(Y_train2_class.values).long()

X_test = torch.Tensor(X_test2.values).float()
Y_test = torch.Tensor(Y_test2_class.values).long()


# defining neural network
net = Classification(input_neurons, hidden_neurons, output_neurons)

# defining loss function
loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

criterionBimodal = nn.CrossEntropyLoss(reduction='none')
# defining optimiser function
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    # Performing forward pass
    Y_pred = net(X)
    
    # Calculating loss 
    loss = loss_func(Y_pred, Y)
    
    # for calculating loss at epoch 10
    if(epoch==10):
        lossbdr = criterionBimodal(Y_pred, Y)
        lossat0 = lossbdr.detach().numpy()
        
    # for calculating loss at epoch 50
    if(epoch==50):
        lossbdr100 = criterionBimodal(Y_pred, Y)
        lossat100 = lossbdr100.detach().numpy()
        
    if((epoch+1)%50 == 0):
        lossbdr = criterionBimodal(Y_pred, Y)
        # Checking if pattern loss > mean loss
        indexes = lossbdr>loss.mean()
        subset = ((indexes==1).nonzero()).squeeze()
        inversesubset = ((indexes==0).nonzero()).squeeze()
        
        # Creating new tensors and removing the patterns with loss > mean loss in X and Y
        X_bdr = X[subset.data]
        X = X[inversesubset.data]
        Y_bdr = Y[subset.data]
        Y = Y[inversesubset.data]
        
        #Creating tensor for loss in the subse
        error_subset = lossbdr[subset.data]
        X_bdr_mean = error_subset
        
        # Calculating variance
        variance = np.var(X_bdr_mean.data.numpy())
        
        #Stopping the training if variance <0.001 and putting the remaining subset data to X
        if(variance<0.001):
            X = torch.cat((X, X_bdr), dim=0)
            Y = torch.cat((Y, Y_bdr), dim=0)
            break
        #Calculating standard deviation
        X_bdr_std = error_subset.std()
        
        # getting the indexes where error > mean + alpha*standard deviation
        X_bdr_sum = X_bdr_mean+(X_bdr_std*0.3)
        final_index = X_bdr_mean.mean()>=X_bdr_sum
        
        #creatin final subset
        final_subset = ((final_index==0).nonzero()).squeeze();
        X_final = X_bdr[final_subset.data]
        Y_final = Y_bdr[final_subset.data]
        
        # adding the remaining data back to X
        X = torch.cat((X, X_final), dim=0)
        Y = torch.cat((Y, Y_final), dim=0)
        
        removed = ((final_index==1).nonzero()).squeeze();
        count = list(removed.size())
        
        print("Removed Outliers ", count)
    
    endbdr = time.time()
    all_losses.append(loss.item())
    if ((epoch+1) % 10 == 0):
        _, predicted = torch.max(Y_pred, 1)
        print('Epoch [%d/%d] Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    net.zero_grad()
    
    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimiser.step()

# Showing pattern error graph at epoch 10 
plt.figure()
plt.hist(lossat0)
plt.show()

# Showing pattern error graph at epoch 50 
plt.figure()
plt.hist(lossat100)
plt.show()

# Showing Graph of all losses
plt.figure()
plt.plot(all_losses)
plt.show()

#Creating confusion matrix
confusion = torch.zeros(output_neurons, output_neurons)

# Getting training accuracy
Y_pred = net(X)
_, predicted = torch.max(Y_pred, 1)
total = predicted.size(0)
correct = sum(predicted.data.numpy() == Y.data.numpy())
print('Training Accuracy: %.2f %%' % (100 * correct / total))

for i in range(X.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

# getting predictions for X test
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
confusion_test = torch.zeros(output_neurons, output_neurons)
for i in range(X_test2.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for testing:')
print(confusion_test)

# predicting FLAG for Oil Well 3
X = torch.Tensor(X_train3.values).float()
Y = torch.Tensor(Y_train3_class.values).long()

X_test = torch.Tensor(X_test3.values).float()
Y_test = torch.Tensor(Y_test3_class.values).long()

# defining neural network
net = Classification(input_neurons, hidden_neurons, output_neurons)

# defining loss function
loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

criterionBimodal = nn.CrossEntropyLoss(reduction='none')
# defining optimiser function
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# for holding losses
all_losses = []

for epoch in range(num_epochs):
    # Performing forward pass
    Y_pred = net(X)
    
    # Calculating loss 
    loss = loss_func(Y_pred, Y)
    
    # for calculating loss at epoch 10
    if(epoch==10):
        lossbdr = criterionBimodal(Y_pred, Y)
        lossat0 = lossbdr.detach().numpy()
        
    # for calculating loss at epoch 50
    if(epoch==50):
        lossbdr100 = criterionBimodal(Y_pred, Y)
        lossat100 = lossbdr100.detach().numpy()
        
    if((epoch+1)%50 == 0):
        lossbdr = criterionBimodal(Y_pred, Y)
        # Checking if pattern loss > mean loss
        indexes = lossbdr>loss.mean()
        subset = ((indexes==1).nonzero()).squeeze()
        inversesubset = ((indexes==0).nonzero()).squeeze()
        
        # Creating new tensors and removing the patterns with loss > mean loss in X and Y
        X_bdr = X[subset.data]
        X = X[inversesubset.data]
        Y_bdr = Y[subset.data]
        Y = Y[inversesubset.data]
        
        #Creating tensor for loss in the subse
        error_subset = lossbdr[subset.data]
        X_bdr_mean = error_subset
        
        # Calculating variance
        variance = np.var(X_bdr_mean.data.numpy())
        
        #Stopping the training if variance <0.001 and putting the remaining subset data to X
        if(variance<0.001):
            X = torch.cat((X, X_bdr), dim=0)
            Y = torch.cat((Y, Y_bdr), dim=0)
            break
        #Calculating standard deviation
        X_bdr_std = error_subset.std()
        
        # getting the indexes where error > mean + alpha*standard deviation
        X_bdr_sum = X_bdr_mean+(X_bdr_std*0.3)
        final_index = X_bdr_mean.mean()>=X_bdr_sum
        
        #creatin final subset
        final_subset = ((final_index==0).nonzero()).squeeze();
        X_final = X_bdr[final_subset.data]
        Y_final = Y_bdr[final_subset.data]
        
        # adding the remaining data back to X
        if(X_final.dim()!=1 and Y_final.dim()!=1):
            X = torch.cat((X, X_final), dim=0)
            Y = torch.cat((Y, Y_final), dim=0)
        
        removed = ((final_index==1).nonzero()).squeeze();
        count = list(removed.size())
        
        print("Removed Outliers ", count)
    
    endbdr = time.time()
    all_losses.append(loss.item())
    if ((epoch+1) % 10 == 0):
        _, predicted = torch.max(Y_pred, 1)
        print('Epoch [%d/%d] Loss: %.4f'
              % (epoch + 1, num_epochs, loss.item()))

    # clearing the gradient before running backward pass
    net.zero_grad()
    
    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimiser.step()

# Showing pattern error graph at epoch 10 
plt.figure()
plt.hist(lossat0)
plt.show()

# Showing pattern error graph at epoch 50 
plt.figure()
plt.hist(lossat100)
plt.show()

# Showing Graph of all losses
plt.figure()
plt.plot(all_losses)
plt.show()

#Creating confusion matrix
confusion = torch.zeros(output_neurons, output_neurons)

# Getting training accuracy
Y_pred = net(X)
_, predicted = torch.max(Y_pred, 1)
total = predicted.size(0)
correct = sum(predicted.data.numpy() == Y.data.numpy())
print('Training Accuracy: %.2f %%' % (100 * correct / total))

for i in range(X.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

# getting predictions for X test
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
confusion_test = torch.zeros(output_neurons, output_neurons)
for i in range(X_test3.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for testing:')
print(confusion_test)