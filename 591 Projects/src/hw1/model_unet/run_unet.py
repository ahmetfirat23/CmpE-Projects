import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from UNET_model import UNET
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision.transforms.functional import rgb_to_grayscale
import argparse

def display():
    with torch.no_grad():
        unet_output = unet(test_imgs_before, test_acts)
        _, axes = plt.subplots(10, 3, figsize=(6, 15))
        for i in range(10):
            axes[i][0].imshow((test_imgs_before[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cmap="gray" if is_grayscale else None)
            axes[i][1].imshow((unet_output[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cmap="gray" if is_grayscale else None)
            axes[i][2].imshow((test_imgs_after[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cmap="gray" if is_grayscale else None)
            axes[i][0].axis('off')
            axes[i][1].axis('off')
            axes[i][2].axis('off')
        plt.savefig("unet_test.png")
        nn.functional.mse_loss(unet_output, test_imgs_after)
        print(f'mse: {nn.functional.mse_loss(unet_output, test_imgs_after)}')

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--grayscale', action='store_true')
argparser.add_argument('--loss', type=str, default='mse')
argparser.add_argument('--save', type=str, default='unet')
argparser.add_argument('--scheduler', action='store_true')
argparser.add_argument('--load', type=str, default='unet.pth')
args = argparser.parse_args()


EPOCH_NUM = args.epoch
BATCH_SIZE = args.batch_size
LR = args.lr

imgs_before = []
imgs_after = []
acts = []

is_grayscale = args.grayscale
loss_type = args.loss
save_path = args.save

for i in range(10):
    img_before= torch.load(f'data/imgs_before_unet_{i}.pt')
    img_after= torch.load(f'data/imgs_after_unet_{i}.pt')
    act = torch.load(f'data/actions_unet_{i}.pt')
    imgs_before.append(img_before)
    imgs_after.append(img_after)
    acts.append(act)
    print(f'loaded {i}')

imgs_before = torch.cat(imgs_before, 0)
imgs_before = imgs_before.float() / 255.0
if is_grayscale:
    imgs_before = rgb_to_grayscale(imgs_before)
imgs_after = torch.cat(imgs_after, 0)

imgs_after = imgs_after.float() / 255.0
if is_grayscale:
    imgs_after = rgb_to_grayscale(imgs_after)
acts = torch.cat(acts, 0)
acts = F.one_hot(acts.long(), 4)

test_size = int(imgs_before.size(0) * 0.1)
val_size = int(imgs_before.size(0) * 0.1)
train_size = imgs_before.size(0) - test_size - val_size

train_imgs_before = imgs_before[:train_size]
train_imgs_after = imgs_after[:train_size]
train_acts = acts[:train_size]

val_imgs_before = imgs_before[train_size:train_size + val_size]
val_imgs_after = imgs_after[train_size:train_size + val_size]
val_acts = acts[train_size:train_size + val_size]

test_imgs_before = imgs_before[train_size + val_size:]
test_imgs_after = imgs_after[train_size + val_size:]
test_acts = acts[train_size + val_size:]

batch_count =  np.ceil(train_imgs_before.size(0) / BATCH_SIZE).astype(int)

unet = UNET(is_grayscale)

if is_grayscale:
    summary(unet, input_size=[(BATCH_SIZE, 1, 128, 128), (BATCH_SIZE, 4)])
else:
    summary(unet, input_size=[(BATCH_SIZE, 3, 128, 128), (BATCH_SIZE, 4)])

if loss_type == 'mse':
    criterion = nn.MSELoss()
elif loss_type == 'l1':
    criterion = nn.L1Loss()
else:
    raise Exception(f'loss {loss_type} not supported')

optimizer_unet = optim.Adam(unet.parameters(), lr=LR)
if args.scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer_unet, step_size=50, gamma=0.5)

unet_losses = []
unet_val_losses = []
try:
    unet.load_state_dict(torch.load(f'{save_path}/{args.load}'))
    try:
        epoch_shift = int(args.load.split('_')[2].split('.')[0])
    except:
        epoch_shift = 200
except:
    epoch_shift = 0

for epoch in (pbar := tqdm(range(EPOCH_NUM))):
    epoch = epoch + epoch_shift
    running_loss = 0.0
    unet.train()
    for i in tqdm(range(batch_count)):
        optimizer_unet.zero_grad()
        batch_imgs_before = train_imgs_before[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_imgs_after = train_imgs_after[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_acts = train_acts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        outputs = unet(batch_imgs_before, batch_acts)
        loss = criterion(outputs.flatten(start_dim=1), batch_imgs_after.flatten(start_dim=1))
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)
        optimizer_unet.step()

        running_loss += loss.item()

        if np.isnan(running_loss):
            raise Exception(f"nan error on batch {i} epoch {epoch}")
        
    unet_losses.append(running_loss / batch_count)
    unet.eval()

    if (epoch <= 100 and epoch % 50 == 0) or (epoch > 100 and epoch % 10 == 0):
        with torch.no_grad():
            outputs = unet(val_imgs_before, val_acts)
            loss = criterion(outputs.flatten(start_dim=1), val_imgs_after.flatten(start_dim=1))
            unet_val_losses.append(loss.item())
        torch.save(unet.state_dict(), f'{save_path}/unet_epoch_{epoch}.pth')
    pbar.set_postfix({'loss': unet_losses[-1], 'val_loss': unet_val_losses[-1]})
    if args.scheduler:
        scheduler.step()

# Save the model
torch.save(unet.state_dict(), f'{save_path}/unet.pth')

np.save(f'{save_path}/unet_losses.npy', unet_losses)
np.save(f'{save_path}/unet_val_losses.npy', unet_val_losses)
periods = list(range(0,200))
plt.plot(periods, np.log(unet_losses), label='train')
periods = list(range(0, 100, 50)) + list(range(100, 200, 10))
plt.plot(periods, np.log(unet_val_losses), label='val')

plt.legend()
plt.savefig(f"{save_path}/unet_loss.png")

display()