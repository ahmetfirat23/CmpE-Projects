import torch
from CNN_model import CNN
from MLP_model import MLP
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision.transforms.functional import rgb_to_grayscale

EPOCH_NUM = 300
BATCH_SIZE = 32
 
device = "cuda:1" if torch.cuda.is_available() else "cpu"

poss = []
acts = []
imgs = []
for i in range(10):
    pos = torch.load(f'data/positions_{i}.pt')
    act = torch.load(f'data/actions_{i}.pt')
    img = torch.load(f'data/imgs_{i}.pt')
    poss.append(pos)
    acts.append(act)
    imgs.append(img)
    print(f'loaded {i}')
pos = torch.cat(poss, 0)
pos = pos.to(device)
acts = torch.cat(acts, 0)
acts = acts.to(device)
acts = F.one_hot(acts.long(), 4)
imgs = torch.cat(imgs, 0)
imgs = imgs.to(device)

print(imgs.device)

imgs = imgs.float() / 255.0
imgs = rgb_to_grayscale(imgs)

test_size = int(imgs.size(0) * 0.2)
val_size = int(imgs.size(0) * 0.2)
train_size = imgs.size(0) - test_size - val_size

train_imgs = imgs[:train_size]
train_acts = acts[:train_size]
train_pos = pos[:train_size]

val_imgs = imgs[train_size:train_size + val_size]
val_acts = acts[train_size:train_size + val_size]
val_pos = pos[train_size:train_size + val_size]

test_imgs = imgs[train_size + val_size:]
test_acts = acts[train_size + val_size:]
test_pos = pos[train_size + val_size:]

batch_count =  np.ceil(train_imgs.size(0) / BATCH_SIZE).astype(int)

cnn = CNN().to(device)
mlp = MLP().to(device)

summary(cnn, input_size=[(BATCH_SIZE, 1, 128, 128), (BATCH_SIZE, 4)])
summary(mlp, input_size=[(BATCH_SIZE, 1, 128, 128), (BATCH_SIZE, 4)])

criterion = nn.MSELoss()
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)
scheduler_cnn = optim.lr_scheduler.StepLR(optimizer_cnn, step_size=50, gamma=0.5)
optimizer_mlp = optim.Adam(mlp.parameters(), lr=0.001)
scheduler_mlp = optim.lr_scheduler.StepLR(optimizer_mlp, step_size=50, gamma=0.5)

cnn_losses = []
cnn_val_losses = []
for epoch in (pbar := tqdm(range(EPOCH_NUM))):
    running_loss = 0.0
    cnn.train()
    for i in range(batch_count):
        inputs = imgs[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        actions = acts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        labels = pos[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        inputs = inputs.float()
        optimizer_cnn.zero_grad()
        outputs = cnn(inputs, actions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss += loss.item()

    cnn.eval()
    val_outputs = cnn(val_imgs.float(), val_acts.float())
    val_loss = criterion(val_outputs, val_pos)

    cnn_losses.append(running_loss / batch_count)
    cnn_val_losses.append(val_loss.item())

    pbar.set_description(f'epoch {epoch+1}/{EPOCH_NUM} loss: {running_loss / batch_count :.4f} val_loss: {val_loss.item():.4f}')

    scheduler_cnn.step()

mlp_losses = []
mlp_val_losses = []
for epoch in (pbar := tqdm(range(EPOCH_NUM))):
    running_loss = 0.0
    mlp.train()
    for i in range(batch_count):
        inputs = imgs[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        actions = acts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        labels = pos[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        inputs = inputs.float()
        optimizer_mlp.zero_grad()
        outputs = mlp(inputs, actions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_mlp.step()
        running_loss += loss.item()

    mlp.eval()
    val_outputs = mlp(val_imgs.float(), val_acts.float())
    val_loss = criterion(val_outputs, val_pos)

    mlp_losses.append(running_loss / batch_count)
    mlp_val_losses.append(val_loss.item())

    pbar.set_description(f'epoch {epoch+1}/{EPOCH_NUM} loss: {running_loss / batch_count :.4f} val_loss: {val_loss.item():.4f}')

    scheduler_mlp.step()

print('Finished Training')

plt.plot(np.log(cnn_losses[1:]), label='cnn')
plt.plot(np.log(mlp_losses[1:]), label='mlp')
plt.legend()
plt.savefig('loss.png')

cnn_output = cnn(test_imgs.float(), test_acts.float())
mlp_output = mlp(test_imgs.float(), test_acts.float())
cnn_loss = criterion(cnn_output, test_pos)
mlp_loss = criterion(mlp_output, test_pos)
print(f'cnn test loss: {cnn_loss.item()}')
print(f'mlp test loss: {mlp_loss.item()}')


torch.save(cnn.state_dict(), 'cnn.pth')
torch.save(mlp.state_dict(), 'mlp.pth')
