#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:55:35 2019

@author: abhishek
"""

import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import torch.nn.functional as F

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

# Selecting the last column for Y
Y_train1_class =  ow1_train.iloc[:, 10]
Y_test1_class = ow1_test.iloc[:, 10]
Y_train2_class =  ow2_train.iloc[:, 10]
Y_test2_class = ow2_test.iloc[:, 10]
Y_train3_class =  ow3_train.iloc[:, 10]
Y_test3_class = ow3_test.iloc[:, 10]

# Setting parameters for Neural Network
input_neurons = 8
hidden_neurons = 50
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

# defining optimiser function
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# for holding losses
all_losses = []


for epoch in range(num_epochs):
    # Performing forward pass
    Y_pred = net(X)
    
    # Calculating loss
    loss = loss_func(Y_pred, Y)
    
    all_losses.append(loss.item())
    if ((epoch+1) % 10 == 0):
        _, predicted = torch.max(Y_pred, 1)
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()
        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct)/total))

    # clearing the gradient before running backward pass
    net.zero_grad()
    
    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimiser.step()

# plotting the losses 
plt.figure()
plt.plot(all_losses)
plt.show()

#Creating confusion matric
confusion = torch.zeros(output_neurons, output_neurons)
Y_pred = net(X)
_, predicted = torch.max(Y_pred, 1)

for i in range(X_train1.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

# Get predictions for test data
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

# Printing Testing accuracy
print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
confusion_test = torch.zeros(output_neurons, output_neurons)
for i in range(X_test1.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for testing:')
print(confusion_test)

# Predicting FLAG for dataset 2

X = torch.Tensor(X_train2.values).float()
Y = torch.Tensor(Y_train2_class.values).long()
X_test = torch.Tensor(X_test2.values).float()
Y_test = torch.Tensor(Y_test2_class.values).long()
# defining neural network
net = Classification(input_neurons, hidden_neurons, output_neurons)

# defining loss function
loss_func = torch.nn.CrossEntropyLoss()

# defining optimiser function
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# for holding losses
all_losses = []


for epoch in range(num_epochs):
    # Performing forward pass
    Y_pred = net(X)
    
    # Calculating loss
    loss = loss_func(Y_pred, Y)
    
    all_losses.append(loss.item())
    if ((epoch+1) % 10 == 0):
        _, predicted = torch.max(Y_pred, 1)
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()
        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct)/total))

    # clearing the gradient before running backward pass
    net.zero_grad()
    
    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimiser.step()

# plotting the losses 
plt.figure()
plt.plot(all_losses)
plt.show()

#Creating confusion matric
confusion = torch.zeros(output_neurons, output_neurons)
Y_pred = net(X)
_, predicted = torch.max(Y_pred, 1)

for i in range(X_train2.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

# Get predictions for test data
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

# Printing Testing accuracy
print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
confusion_test = torch.zeros(output_neurons, output_neurons)
for i in range(X_test2.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for testing:')
print(confusion_test)

# Predicting FLAG for datassest 3

X = torch.Tensor(X_train3.values).float()
Y = torch.Tensor(Y_train3_class.values).long()
X_test = torch.Tensor(X_test3.values).float()
Y_test = torch.Tensor(Y_test3_class.values).long()

# defining neural network
net = Classification(input_neurons, hidden_neurons, output_neurons)

# defining loss function
loss_func = torch.nn.CrossEntropyLoss()

# defining optimiser function
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# for holding losses
all_losses = []


for epoch in range(num_epochs):
    # Performing forward pass
    Y_pred = net(X)
    
    # Calculating loss
    loss = loss_func(Y_pred, Y)
    
    all_losses.append(loss.item())
    if ((epoch+1) % 10 == 0):
        _, predicted = torch.max(Y_pred, 1)
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()
        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct)/total))

    # clearing the gradient before running backward pass
    net.zero_grad()
    
    # performing backward pass
    loss.backward()
    
    #using optimzer to optimize the parameters
    optimiser.step()

# plotting the losses 
plt.figure()
plt.plot(all_losses)
plt.show()

#Creating confusion matric
confusion = torch.zeros(output_neurons, output_neurons)
Y_pred = net(X)
_, predicted = torch.max(Y_pred, 1)

for i in range(X_train3.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

# Get predictions for test data
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

# Printing Testing accuracy
print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
confusion_test = torch.zeros(output_neurons, output_neurons)
for i in range(X_test3.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for testing:')
print(confusion_test)


