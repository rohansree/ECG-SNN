import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from data_extract import extract_features
from model import LIFNeuralNetwork 

directory_path = 'mit-bih-arrhythmia-database-1.0.0/'

# get all record paths
record_paths = glob.glob(os.path.join(directory_path, '*.dat'))
record_paths = [path.replace('.dat', '') for path in record_paths]

# for indiv. analysis later (each class prediction rates)
accuracy_rates = []
predicted_class_rates = {cls: [] for cls in ['N', 'S', 'V', 'F', 'Q']}
actual_class_rates = {cls: [] for cls in ['N', 'S', 'V', 'F', 'Q']}
file_names = []

#one loop is one file in the MIT-BIH database
for record_path in record_paths:
    #each iteration of the loop is like one run_single.py
    annotation_path = record_path
    
    seq1, seq2, labels = extract_features(record_path, annotation_path)
    
    #80:20 train:test split
    X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(seq1, seq2, labels, test_size=0.2, shuffle=True, random_state=42)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(['N', 'S', 'V', 'F', 'Q'])
    
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = LIFNeuralNetwork() 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) #learning rate change is here
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs1.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
    
    predicted_labels_encoded = []
    actual_labels_encoded = []
    
    model.eval()
    with torch.no_grad():
        for inputs1, inputs2, labels in test_loader:
            outputs = model(inputs1, inputs2)
            _, predicted = torch.max(outputs, 1)
            predicted_labels_encoded.extend(predicted.tolist())
            actual_labels_encoded.extend(labels.tolist())
    
    predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
    actual_labels = label_encoder.inverse_transform(actual_labels_encoded)
    
    # calculate accuracy and non-normal metrics
    total_labels = len(predicted_labels)
    non_N_labels = np.sum(predicted_labels != 'N')
    percentage_non_N = (non_N_labels / total_labels) * 100
    
    total_labels_a = len(actual_labels)
    non_N_labels_a = np.sum(actual_labels != 'N')
    percentage_non_N_a = (non_N_labels_a / total_labels_a) * 100
    
    differences = predicted_labels != actual_labels
    num_differences = np.sum(differences)
    percentage_different = (num_differences / total_labels) * 100
    percentage_same = 100 - percentage_different
    
    # store metrics (for this ONE file) in the running total lists
    accuracy_rates.append(percentage_same)
    # predicted_non_N_rates.append(percentage_non_N)
    # actual_non_N_rates.append(percentage_non_N_a)
    file_names.append(os.path.basename(record_path)) #already visited files

    # calculate percentage of each class in predicted and actual labels
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        predicted_cls_count = np.sum(predicted_labels == cls)
        actual_cls_count = np.sum(actual_labels == cls)
        predicted_cls_rate = (predicted_cls_count / total_labels) * 100
        actual_cls_rate = (actual_cls_count / total_labels) * 100
        predicted_class_rates[cls].append(predicted_cls_rate)
        actual_class_rates[cls].append(actual_cls_rate)
        #append to the running metrics
    
    
    # Print current file metrics
    # print(f"Processed {os.path.basename(record_path)} - Accuracy: {percentage_same:.2f}%, Predicted Non-N: {percentage_non_N:.2f}%, Actual Non-N: {percentage_non_N_a:.2f}%")
    print(f"Processed {os.path.basename(record_path)} - Accuracy: {percentage_same:.2f}%")

#including non-N rates

# plt.figure(figsize=(14, 7))

# plt.subplot(1, 2, 1)
# plt.plot(file_names, accuracy_rates, marker='o', linestyle='-')
# plt.xticks(rotation=90)
# plt.xlabel('File')
# plt.ylabel('Accuracy Rate (%)')
# plt.title('Model Accuracy Rate by File')

# plt.subplot(1, 2, 2)
# plt.plot(file_names, predicted_non_N_rates, marker='o', linestyle='-', label='Predicted Non-N %')
# plt.plot(file_names, actual_non_N_rates, marker='x', linestyle='--', label='Actual Non-N %')
# plt.xticks(rotation=90)
# plt.xlabel('File')
# plt.ylabel('Non-N Rate (%)')
# plt.title('Non-N Rate by File')
# plt.legend()

# plt.tight_layout()
# plt.show()

# just accuracy
plt.figure(figsize=(12, 6))

plt.plot(file_names, accuracy_rates, label='Accuracy Rate', marker='o')

#for each class
# for cls in ['N', 'S', 'V', 'F', 'Q']:
#     plt.plot(file_names, predicted_class_rates[cls], label=f'Predicted {cls} Rate', linestyle='--')
#     plt.plot(file_names, actual_class_rates[cls], label=f'Actual {cls} Rate', linestyle='-')

plt.xlabel('File')
plt.ylabel('Percentage')
plt.title('Model Performance by File')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()