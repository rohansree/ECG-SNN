from sklearn.calibration import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from data_extract import extract_features

from model_cuda import LIFNeuralNetwork

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import wfdb
import numpy as np
import matplotlib.pyplot as plt

import time

# Start profiling
overall_start_time = time.time()


#path of the SINGLE file
annotation_path = '../ECG-SNN/mit-bih-arrhythmia-database-1.0.0/105'
# mit-bih-arrhythmia-database-1.0.0
record_path = '../ECG-SNN/mit-bih-arrhythmia-database-1.0.0/105'

#pull the signals and corresponding labels from the file
seq1, seq2, labels = extract_features(record_path, annotation_path)

# split the data into training and testing sets
# 80:20 train:test split
X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(seq1, seq2, labels, test_size=0.2, shuffle=True, random_state=42)

#in case the train doesn't have all the labels
label_encoder = LabelEncoder()
label_encoder.fit(['N', 'S', 'V', 'F', 'Q'])

# each class encoded to a number
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# make each number label aa tensor so pytorch can be used
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

#the data sets get converted to np arrays and then tensors
#and then to pytorch datasets and data loaders

#for efficiency (python gives warnings for slow speed otherwise)
X_train1_combined = np.array(X_train1)
X_test1_combined = np.array(X_test1)
X_train2_combined = np.array(X_train2)
X_test2_combined = np.array(X_test2)

X_train_tensor1 = torch.tensor(X_train1_combined, dtype=torch.float32)
X_test_tensor1 = torch.tensor(X_test1_combined, dtype=torch.float32)
X_train_tensor2 = torch.tensor(X_train2_combined, dtype=torch.float32)
X_test_tensor2 = torch.tensor(X_test2_combined, dtype=torch.float32)


train_dataset = TensorDataset(X_train_tensor1, X_train_tensor2, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor1, X_test_tensor2, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# initialize model and loss function
# model = NeuralNetwork()
with torch.no_grad():
    model = LIFNeuralNetwork().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

device = torch.device("cuda")


# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs1, inputs2, labels in train_loader:

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs1.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

predicted_labels_encoded = []
actual_labels_encoded = []
model.eval()
with torch.no_grad():
    for inputs1, inputs2, labels in test_loader:

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = labels.to(device)

        outputs = model(inputs1, inputs2)
        _, predicted = torch.max(outputs, 1)
        predicted_labels_encoded.extend(predicted.tolist())
        actual_labels_encoded.extend(labels.tolist())

predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
actual_labels = label_encoder.inverse_transform(actual_labels_encoded)


total_labels = len(predicted_labels)

# count  number of predicted labels that are not equal to N
non_N_labels = np.sum(predicted_labels != 'N')

# calculate percentage of non-N predicted labels
percentage_non_1 = (non_N_labels / total_labels) * 100


print(f"Predicted percentage of labels that are not N: {percentage_non_1:.2f}%")

#more than 10% non-normal predicted
if percentage_non_1 > 10.0:
    print("Model predicts significant abnormality")

total_labels_a = len(actual_labels)

# count number of actual labels that are not equal to N
non_N_labels_a = np.sum(actual_labels != 'N')

# print(total_labels_a)
# print(non_N_labels_a)

# calculate percentage of non-N actual labels
percentage_non_1_a = (non_N_labels_a / total_labels_a) * 100

print(f"Actual Percentage of labels that are not N: {percentage_non_1_a:.2f}%")

# Straightforward: see how many predictions were wrong
differences = predicted_labels != actual_labels
num_differences = np.sum(differences)

# See how many predictions were wrong as a percentage
total_elements = len(predicted_labels)
percentage_different = (num_differences / total_elements) * 100
percentage_same = 100 - percentage_different

print(f"Model Accuracy: {percentage_same:.2f}%")

overall_end_time = time.time()
print(f"Total Execution Time: {overall_end_time - overall_start_time:.2f} seconds")

#----------------------- Extra analysis: By individual class -----------------------

unique_classes = np.unique(actual_labels)

# # confusion matrix
# conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=unique_classes)
# print("Confusion Matrix:")
# print(conf_matrix)

# classification report
# class_report = classification_report(actual_labels, predicted_labels, target_names=unique_classes)
# print("Classification Report:")
# print(class_report)

# unique_labels, predicted_counts = np.unique(predicted_labels, return_counts=True)
# unique_labels_a, actual_counts = np.unique(actual_labels, return_counts=True)

# # print("Predicted label counts:")
# for label, count in zip(unique_labels, predicted_counts):
#     print(f"{label}: {count}")

# print("Actual label counts:")
# for label, count in zip(unique_labels_a, actual_counts):
#     print(f"{label}: {count}")
