import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
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
import pickle

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
    
    def load_weights(self, weights):
        self.fc1.weight = nn.Parameter(torch.tensor(weights[0], dtype=torch.float32))
        self.fc1.bias = nn.Parameter(torch.tensor(weights[1].flatten(), dtype=torch.float32))
        self.fc2.weight = nn.Parameter(torch.tensor(weights[2], dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor(weights[3].flatten(), dtype=torch.float32))
        self.fc3.weight = nn.Parameter(torch.tensor(weights[4], dtype=torch.float32))
        self.fc3.bias = nn.Parameter(torch.tensor(weights[5].flatten(), dtype=torch.float32))

    def save_weights(self):
        w1 = self.fc1.weight.detach().numpy()
        b1 = self.fc1.bias.detach().numpy().reshape(-1, 1)
        w2 = self.fc2.weight.detach().numpy()
        b2 = self.fc2.bias.detach().numpy().reshape(-1, 1)
        w3 = self.fc3.weight.detach().numpy()
        b3 = self.fc3.bias.detach().numpy().reshape(-1, 1)
        return [w1, b1, w2, b2, w3, b3]

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.save_weights(), f)

    def save_model_from_weights(self, weights, filename):
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)
        print("Model weights saved successfully.")
        return weights


    def load_model(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        self.load_weights(weights)
        print("Model loaded successfully.")
        return weights

def train(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10, save_best=True):
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_model_weights = None
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(running_corrects/len(train_loader.dataset))
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels)
        val_losses.append(val_running_loss/len(validation_loader))
        val_accuracies.append(val_running_corrects/len(validation_loader.dataset))

        if save_best and val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_weights = model.save_weights()
            best_epoch = epoch
        
        tqdm.write(f'Epoch {epoch+1}, Loss: {train_losses[-1]}, Accuracy: {train_accuracies[-1]}')
        tqdm.write(f'Validation Loss: {val_losses[-1]}, Validation Accuracy: {val_accuracies[-1]}')

    if save_best:
        model.save_model_from_weights(best_model_weights, 'best_torch_model.pkl')
        print(f"Best model saved at epoch {best_epoch+1} with validation loss: {best_val_loss} and accuracy: {val_accuracies[best_epoch]}")
    
    model.save_model('final_torch_model.pkl')
    print("Final model saved successfully.")
    return train_losses, val_losses, train_accuracies, val_accuracies

def test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(outputs.numpy())
    return all_preds, all_labels, all_probs


train_images = np.load('quickdraw_subset_np/train_images.npy')
train_labels = np.load('quickdraw_subset_np/train_labels.npy')
test_images = np.load('quickdraw_subset_np/test_images.npy')
test_labels = np.load('quickdraw_subset_np/test_labels.npy')

print(train_images.shape) # (20000, 28, 28)
print(test_images.shape) # (5000, 28, 28)

train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0

val_images = train_images[:2000]
val_labels = train_labels[:2000]
train_images = train_images[2000:]
train_labels = train_labels[2000:]

print(train_images.shape) # (20000, 784)
print(test_images.shape) # (5000, 784)

print(np.unique(val_labels, return_counts=True))
print(np.unique(train_labels, return_counts=True))
print(np.unique(test_labels, return_counts=True))

train_loader = DataLoader(list(zip(train_images, train_labels)), batch_size=64, shuffle=True)
validation_loader = DataLoader(list(zip(val_images, val_labels)), batch_size=64, shuffle=False)
test_loader = DataLoader(list(zip(test_images, test_labels)), batch_size=64, shuffle=False)


model = MLP(784, 128, 5)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

history = train(model, train_loader, validation_loader, criterion, optimizer, num_epochs=100)

model.load_model('best_torch_model.pkl')

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


all_preds, all_labels, all_probs = test(model, test_loader)

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
