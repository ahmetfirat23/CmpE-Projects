import numpy as np
import utils
from tqdm import tqdm
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

train_images = np.load('quickdraw_subset_np/train_images.npy')
train_labels = np.load('quickdraw_subset_np/train_labels.npy')
test_images = np.load('quickdraw_subset_np/test_images.npy')
test_labels = np.load('quickdraw_subset_np/test_labels.npy')

# reshape the images
train_images = train_images.reshape(-1, 784).astype(np.float64) / 255.0
test_images = test_images.reshape(-1, 784).astype(np.float64) / 255.0

# create train and val splits
val_images = train_images[:2000]
val_labels = train_labels[:2000]
train_images = train_images[2000:]
train_labels = train_labels[2000:]

# define data loader
train_loader = utils.DataLoader(train_images, train_labels, batch_size=64, shuffle=True)
validation_loader = utils.DataLoader(val_images, val_labels, batch_size=64, shuffle=False)
test_loader = utils.DataLoader(test_images, test_labels, batch_size=64, shuffle=False)

# mlp for image embedding
class MLP_image(utils.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = utils.Linear(input_size, hidden_size)
        self.relu1 = utils.ReLU()
        self.fc2 = utils.Linear(hidden_size, hidden_size)
        self.relu2 = utils.ReLU()
        self.fc3 = utils.Linear(hidden_size, output_size)
        self.construct() # no softmax here.
    
    def construct(self):
        self.cache.append(self.fc1)
        self.cache.append(self.relu1)
        self.cache.append(self.fc2)
        self.cache.append(self.relu2)
        self.cache.append(self.fc3)

# mlp for word embedding.
class MLP_category(utils.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = utils.Linear(input_size, hidden_size)
        self.relu1 = utils.ReLU()
        self.fc2 = utils.Linear(hidden_size, output_size)
        self.construct() # no softmax here.
    
    def construct(self):
        self.cache.append(self.fc1)
        self.cache.append(self.relu1)
        self.cache.append(self.fc2)

model_img = MLP_image(784, 512, 32) # 3 layered
model_text = MLP_category(100, 32, 32) # 2 layered

criterion = utils.CosineSimilarityLoss()
optimizer = utils.SGD(model_img, lr=1e-1, momentum=0.9)


# load glove embeddings.
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

label_dict = {0: "rabbit", 1: "yoga", 2: "hand", 3: "snowman", 4: "motorbike"}
labels_sorted = ["rabbit", "yoga", "hand", "snowman", "motorbike"]

# cosine similarity function to be used in class prediction.
def cos_sim(a, b):
    eps = 1e-12
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + eps  # (N, 1)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + eps  # (N, 1)
    
    # Normalize to unit vectors
    a_hat = a / a_norm
    b_hat = b / b_norm

    # Element-wise product and sum across features
    cos_sim_s = np.sum(a_hat * b_hat, axis=1, keepdims=True)  # (N, 1)

    return cos_sim_s

# main train function
def train(model_img, model_text, train_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_size = len(train_loader)
    val_size = len(validation_loader)

    # get the embedding values for the classes.
    embed_values_list = np.stack([glove[label_str] for label_str in labels_sorted])

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            label_strings = [label_dict[x] for x in labels]
            embed_values = np.stack([glove[label_str] for label_str in label_strings])
            
            optimizer.zero_grad()
            
            # get image and word embeddings.
            outputs_1 = model_img(inputs) # embed img
            outputs_2 = model_text(embed_values) # embed cat
            
            # get the loss and take a step. normalization is done inside.
            loss = criterion(outputs_1, outputs_2)
            grad1, grad2 = criterion.backward(outputs_1, outputs_2)
            optimizer.step_w_grad(grad1)

            running_loss += np.mean(loss)

            embed_outputs_list = model_text(embed_values_list)
            guessed_labels = []

            # this part is for calculating the accuracy. similarities are calculated for each class's word embedding.
            # the most similar class is chosen as the predicted class. 
            for img_embed in outputs_1:
                # L2 distance to each class embedding
                label_embed = utils.normalize_np(embed_outputs_list, dim=-1)
                img_embed = utils.normalize_np(img_embed, dim=-1)
                img_embed = img_embed.reshape(1, -1)

                dists = cos_sim(img_embed, label_embed)
                closest_class = np.argmax(dists)
                guessed_labels.append(closest_class)

            guessed_labels = np.array(guessed_labels)
            correct_mask = (guessed_labels == labels)
            
            running_corrects += np.sum(correct_mask)

        train_losses.append(running_loss/(i+1))
        train_accuracies.append(running_corrects/train_size)
        
        val_running_loss = 0.0
        val_running_corrects = 0
        # validation accuracy & loss calculation
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            label_strings = [label_dict[x] for x in labels]
            embed_values = np.stack([glove[label_str] for label_str in label_strings])
            outputs_1 = model_img(inputs) # embed img
            outputs_2 = model_text(embed_values) # embed cat
            loss = criterion(outputs_1, outputs_2)
            val_running_loss += np.mean(loss)

            embed_outputs_list = model_text(embed_values_list)
            guessed_labels = []

            for img_embed in outputs_1:
                # L2 distance to each class embedding
                label_embed = utils.normalize_np(embed_outputs_list, dim=-1)
                img_embed = utils.normalize_np(img_embed, dim=-1)
                img_embed = img_embed.reshape(1, -1)

                dists = cos_sim(img_embed, label_embed)
                closest_class = np.argmax(dists)
                guessed_labels.append(closest_class)

            guessed_labels = np.array(guessed_labels) 
            correct_mask = (guessed_labels == labels)
            val_running_corrects += np.sum(correct_mask)
        val_losses.append(val_running_loss/(i+1))
        val_accuracies.append(val_running_corrects/val_size)
            
        
        tqdm.write(f'Epoch {epoch+1}, Loss: {train_losses[-1]}, Accuracy: {train_accuracies[-1]}')
        tqdm.write(f'Validation Loss: {val_losses[-1]}, Validation Accuracy: {val_accuracies[-1]}')

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
    embed_values_list = np.stack([glove[label_str] for label_str in labels_sorted])
    for data in test_loader:
        inputs, labels = data                
        outputs_1 = model_img(inputs) # embed img

        embed_outputs_list = model_text(embed_values_list)
        guessed_labels = []
        probs = []

        for img_embed in outputs_1:
            # L2 distance to each class embedding
            label_embed = utils.normalize_np(embed_outputs_list, dim=-1)
            img_embed = utils.normalize_np(img_embed, dim=-1)
            img_embed = img_embed.reshape(1, 32)

            dists = cos_sim(img_embed, label_embed)
            closest_class = np.argmax(dists)
            guessed_labels.append(closest_class)
            probs.append(utils.normalize_np(dists))

        guessed_labels = np.array(guessed_labels)  
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