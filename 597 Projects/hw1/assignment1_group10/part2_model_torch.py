import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import time
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

# mlp module for image embedding.
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
# mlp module for word embedding.
class MLP_cat(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_cat, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model_img = MLP(784, 512, 32) # mlp for image embedding
model_text = MLP_cat(100, 32, 32) # mlp for word embedding

train_images = np.load('quickdraw_subset_np/train_images.npy')
train_labels = np.load('quickdraw_subset_np/train_labels.npy')
test_images = np.load('quickdraw_subset_np/test_images.npy')
test_labels = np.load('quickdraw_subset_np/test_labels.npy')

# reshape the images
train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0

# create train and val splits
val_images = train_images[:2000]
val_labels = train_labels[:2000]
train_images = train_images[2000:]
train_labels = train_labels[2000:]

# define data loader
train_loader = DataLoader(list(zip(train_images, train_labels)), batch_size=64, shuffle=True)
validation_loader = DataLoader(list(zip(val_images, val_labels)), batch_size=64, shuffle=False)
test_loader = DataLoader(list(zip(test_images, test_labels)), batch_size=64, shuffle=False)

# categorical cross entropy loss
criterion = nn.CosineEmbeddingLoss()
# stochastic gradient descent with momentum
optimizer = optim.SGD(model_img.parameters(), lr=1e-1, momentum=0.9)

label_dict = {0: "rabbit", 1: "yoga", 2: "hand", 3: "snowman", 4: "motorbike"}
labels_sorted = ["rabbit", "yoga", "hand", "snowman", "motorbike"]

# loading glove embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings[word] = vector
    return embeddings

glove_path = 'glove.6B/glove.6B.100d.txt'
glove = load_glove_embeddings(glove_path)
print("Glove embeddings loaded.")

# main train function
def train(model_img, model_text, train_loader, criterion, optimizer, num_epochs=10):
    model_img.train()
    model_text.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # get the embedding values for the classes.
    embed_values_list = torch.tensor([glove[label_str] for label_str in labels_sorted])

    for epoch in tqdm(range(num_epochs)):
        model_img.train()
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            label_strings = [label_dict[x.item()] for x in labels]
            embed_values = torch.tensor([glove[label_str] for label_str in label_strings])
            
            optimizer.zero_grad()

            # get image and word embeddings.
            outputs_1 = model_img(inputs) # embed img
            outputs_2 = model_text(embed_values) # embed cat

            # normalize embeddings.
            outputs_1 = F.normalize(outputs_1, dim=-1)
            outputs_2 = F.normalize(outputs_2, dim=-1)

            # get the loss and take a step.
            loss = criterion(outputs_1, outputs_2, torch.zeros(labels.shape) + 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            embed_outputs_list = model_text(embed_values_list)
            guessed_labels = []

            # this part is for calculating the accuracy. similarities are calculated for each class's word embedding.
            # the most similar class is chosen as the predicted class. 
            for img_embed in outputs_1:
                # L2 distance to each class embedding
                label_embed = F.normalize(embed_outputs_list, dim=-1)
                img_embed = F.normalize(img_embed, dim=-1)
                img_embed = img_embed.reshape(1, -1)

                with torch.no_grad():
                    dists = F.cosine_similarity(img_embed, label_embed, eps = 1e-12)
                closest_class = np.argmax(dists)
                guessed_labels.append(closest_class)

            guessed_labels = np.array(guessed_labels)
            
            running_corrects += sum(torch.from_numpy(guessed_labels) == labels)
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(running_corrects/len(train_loader.dataset))
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {running_corrects/len(train_loader.dataset)}')

        model_img.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        # validation accuracy & loss calculation
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            label_strings = [label_dict[x.item()] for x in labels]
            embed_values = torch.tensor([glove[label_str] for label_str in label_strings])

            outputs_1 = model_img(inputs) # embed img
            outputs_2 = model_text(embed_values) # embed cat
            outputs_1 = F.normalize(outputs_1, dim=-1)
            outputs_2 = F.normalize(outputs_2, dim=-1)

            loss = criterion(outputs_1, outputs_2, torch.zeros(labels.shape) + 1)

            embed_outputs_list = model_text(embed_values_list)
            guessed_labels = []

            for img_embed in outputs_1:
                # L2 distance to each class embedding
                label_embed = F.normalize(embed_outputs_list, dim=-1)
                img_embed = F.normalize(img_embed, dim=-1)
                with torch.no_grad():
                    dists = F.cosine_similarity(img_embed, label_embed, eps = 1e-12) 
                closest_class = np.argmax(dists)
                guessed_labels.append(closest_class)

            val_running_loss += loss.item()
            guessed_labels = np.array(guessed_labels)
            val_running_corrects += sum(torch.from_numpy(guessed_labels) == labels)
            
        val_losses.append(val_running_loss/len(validation_loader))
        val_accuracies.append(val_running_corrects/len(validation_loader.dataset))

        print(f'Validation Loss: {val_running_loss/len(validation_loader)}, Validation Accuracy: {val_running_corrects/len(validation_loader.dataset)}')

    return train_losses, val_losses, train_accuracies, val_accuracies

history = train(model_img, model_text, train_loader, criterion, optimizer, num_epochs=20)

# Plot history graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history[0], label='train')
plt.plot(history[1], label='val')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history[2], label='train')
plt.plot(history[3], label='val')
plt.title('Accuracy')
plt.legend()
plt.show()

# test function for calculating the predictions, labels and probabilities (the similarities in this case.)
def test(model_img, model_text, test_loader):
    all_preds = []
    all_labels = []
    all_probs = []
    embed_values_list = torch.tensor([glove[label_str] for label_str in labels_sorted])
    for data in test_loader:
        inputs, labels = data                
        outputs_1 = model_img(inputs) # embed img

        embed_outputs_list = model_text(embed_values_list)
        guessed_labels = []
        probs = []

        for img_embed in outputs_1:
            # L2 distance to each class embedding
            label_embed = F.normalize(embed_outputs_list, dim=-1)
            img_embed = F.normalize(img_embed, dim=-1)

            with torch.no_grad():
                dists = F.cosine_similarity(img_embed, label_embed, eps = 1e-12)
            closest_class = np.argmax(dists)
            guessed_labels.append(closest_class)
            probs.append(F.normalize(dists, dim=-1))

        guessed_labels = np.array(guessed_labels)  # shape (B,)
        all_probs.extend(probs)

        preds = guessed_labels
        all_preds.extend(preds)
        all_labels.extend(labels)
    return all_preds, all_labels, all_probs

all_preds, all_labels, all_probs = test(model_img, model_text, test_loader)

print('Accuracy:', accuracy_score(all_labels, all_preds))
print('Precision:', precision_score(all_labels, all_preds, average='macro'))
print('Recall:', recall_score(all_labels, all_preds, average='macro'))
print('F1:', f1_score(all_labels, all_preds, average='macro'))

cm = confusion_matrix(all_labels, all_preds)
print(cm)

print(classification_report(all_labels, all_preds))

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
n_classes = 5
all_labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

# ROC curve
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall curve
precision = {}
recall = {}
average_precision = {}

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
    average_precision[i] = average_precision_score(all_labels_bin[:, i], all_probs[:, i])

plt.figure()

for i in range(5):
    plt.plot(recall[i], precision[i], label=f'Precision-Recall curve of class {i} (area = {average_precision[i]:0.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc='lower right')
plt.show()