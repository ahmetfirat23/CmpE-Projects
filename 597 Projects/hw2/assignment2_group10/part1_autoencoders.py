import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

train_images = np.load("quickdraw_subset_np/train_images.npy")  # (N_train, 28, 28)
train_labels = np.load("quickdraw_subset_np/train_labels.npy")  # (N_train,)
test_images = np.load("quickdraw_subset_np/test_images.npy")    # (N_test, 28, 28)
test_labels = np.load("quickdraw_subset_np/test_labels.npy")

# Normalise to [0,1] and add channel dimension
train_images = (train_images.astype(np.float32) / 255.0)[:, None, :, :]
test_images = (test_images.astype(np.float32) / 255.0)[:, None, :, :]

val_split = 0.1
n_total = train_images.shape[0]
idx = int(n_total * (1 - val_split))

val_images, val_labels = train_images[idx:], train_labels[idx:]
train_images, train_labels = train_images[:idx], train_labels[:idx]

train_ds = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels))
val_ds = TensorDataset(torch.from_numpy(val_images), torch.from_numpy(val_labels))
all_ds = TensorDataset(torch.from_numpy(np.concatenate([train_images, test_images], 0)),
                       torch.from_numpy(np.concatenate([train_labels, test_labels], 0)))

BATCH_SIZE = 128
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, latent_dim: int = 64):
        super().__init__()
        self.seq_len = 28
        self.input_dim = 28
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=self.input_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    # encode using lstm
    def encode(self, x):      # (B, 1, 28, 28)
        x_seq = x.squeeze(1)  # (B, 28, 28)
        out, (h_n, _) = self.encoder_lstm(x_seq)
        z = self.fc_enc(h_n[-1])  # (B, latent_dim)
        return z

    # decode using lstm
    def decode(self, z):
        batch_size = z.size(0)
        dec_init = self.fc_dec(z).unsqueeze(1)  # (B, 1, hidden)
        dec_in = dec_init.repeat(1, self.seq_len, 1)  # feed same vector each step
        out, _ = self.decoder_lstm(dec_in)
        out = out.reshape(batch_size, 1, 28, 28)  # reshape to image
        return out

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(64 * 3 * 3, latent_dim) # fc layer after encoder

        self.fc_dec = nn.Linear(latent_dim, 64 * 3 * 3) # fc layer before decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        batch_size = x.size(0)
        h = self.encoder(x)
        h = h.view(batch_size, -1)
        z = self.fc_enc(h)
        return z

    def decode(self, z):
        batch_size = z.size(0)
        h = self.fc_dec(z).view(batch_size, 64, 3, 3)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

# function for training autoencoder
def train_autoencoder(model: nn.Module, train_loader, val_loader, num_epochs: int = 20, lr = 1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        # forward, backward and step for each bach
        for imgs, _ in train_loader:
            optimizer.zero_grad()
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        # validation
        with torch.no_grad():
            for imgs, _ in val_loader:
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                val_loss += loss.item() * imgs.size(0)
        val_loss = val_loss / len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    torch.save(model.state_dict(), f"{model.__class__.__name__}_final_weights.pth")

    return model, history

# function for plotting the loss
def plot_loss(history, title):
    plt.figure()
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.show()

# function for plotting tsne
def tsne_plot(model: nn.Module, dataset: TensorDataset, n_samples: int = 2000, title: str = ""):
    model.eval()
    idx = np.random.choice(len(dataset), size=n_samples, replace=False)
    imgs, labels = zip(*[dataset[i] for i in idx])
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)

    with torch.no_grad():
        _, z = model(imgs)
    z = z.cpu().numpy()
    labels = labels.numpy()

    z_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    plt.title(f"t-SNE of {title} embeddings")
    plt.xticks([])
    plt.yticks([])

    # produce legend with unique labels
    handles, _ = scatter.legend_elements(prop="colors")
    classes = np.unique(labels)
    plt.legend(handles, classes, title="Class", loc="best", fontsize="small")
    plt.show()

if __name__ == "__main__":
    # call lstm
    lstm_ae = LSTMAutoencoder(hidden_dim=128, num_layers=2, latent_dim=64)
    lstm_ae, history_lstm = train_autoencoder(lstm_ae, train_loader, val_loader, num_epochs=60, lr = 5e-3)
    plot_loss(history_lstm, "LSTM Autoencoder MSE")
    tsne_plot(lstm_ae, all_ds, title="LSTM Autoencoder")

    # call conv autoencoder
    conv_ae = ConvAutoencoder(latent_dim=64)
    conv_ae, history_conv = train_autoencoder(conv_ae, train_loader, val_loader, num_epochs=30, lr = 1e-4)
    plot_loss(history_conv, "Convolutional Autoencoder MSE")
    tsne_plot(conv_ae, all_ds, title="Convolutional Autoencoder")
