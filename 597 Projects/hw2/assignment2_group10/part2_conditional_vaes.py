import os
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
import pickle

# --- Utility Functions ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COND_OUTPUT_BASE_DIR = "outputs_conditional"
os.makedirs(COND_OUTPUT_BASE_DIR, exist_ok=True)

def save_image_grid_conditional(images_tensor, filepath, n_rows=4, normalize=True, scale_each=True):
    if images_tensor.ndim == 3: 
        images_tensor = images_tensor.unsqueeze(1)
    grid = make_grid(images_tensor.cpu(), nrow=n_rows, normalize=normalize, scale_each=scale_each, pad_value=0.5)
    save_image(grid, filepath)
    tqdm.write(f"Saved conditional image grid to {filepath}")

# --- VAE Loss ---
def vae_loss_fn(x, x_recon, z_mu, z_logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1).mean()
    return recon_loss + beta * kld, recon_loss, kld

# --- Conditional Models ---
class ConditionalConvVAE(nn.Module):
    def __init__(self,
                 latent_dim, 
                 num_classes,
                 input_channels=1, 
                 label_embedding_dim=32,
                 initial_size=4, 
                 initial_channels_encoder=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        self.initial_size = initial_size
        self.initial_channels_encoder = initial_channels_encoder

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(32, self.initial_channels_encoder, 3, stride=2, padding=1), 
            nn.ReLU(True)
        )
        encoder_flat_dim = self.initial_channels_encoder * self.initial_size * self.initial_size
        self.fc_mu = nn.Linear(encoder_flat_dim + label_embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_flat_dim + label_embedding_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + label_embedding_dim, encoder_flat_dim)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(self.initial_channels_encoder, 32, 3, stride=2, padding=1, output_padding=0), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )

    def encode(self, x, y_idx):
        h = self.encoder_cnn(x).view(x.size(0), -1)
        y_embed = self.label_embedding(y_idx)
        h_cat = torch.cat([h, y_embed], dim=1)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)

    def decode(self, z, y_idx):
        y_embed = self.label_embedding(y_idx)
        z_cat = torch.cat([z, y_embed], dim=1)
        h = self.fc_dec(z_cat).view(z.size(0), self.initial_channels_encoder, self.initial_size, self.initial_size)
        return self.decoder_cnn(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_idx):
        mu, logvar = self.encode(x, y_idx)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y_idx), mu, logvar, z


class ConditionalRNNEncoderConvDecoderVAE(nn.Module):
    def __init__(self, 
                 rnn_type, 
                 input_dim, 
                 hidden_dim_rnn, 
                 num_layers_rnn, 
                 latent_dim, 
                 num_classes,
                 encoder_variant, 
                 bidirectional_encoder,
                 conv_decoder_fc_in_features, 
                 conv_decoder_initial_channels, 
                 conv_decoder_initial_size,
                 label_embedding_dim=32):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.seq_len = input_dim
        self.input_dim_rnn = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        
        self.hidden_dim_rnn = hidden_dim_rnn
        self.num_layers_rnn = num_layers_rnn
        self.encoder_variant = encoder_variant
        self.num_directions = 2 if bidirectional_encoder else 1

        rnn_module = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.encoder_rnn = rnn_module(
            self.input_dim_rnn, hidden_dim_rnn, num_layers_rnn,
            batch_first=True, bidirectional=bidirectional_encoder
        )
        
        encoder_output_fc_dim = hidden_dim_rnn * self.num_directions
        self.fc_mu = nn.Linear(encoder_output_fc_dim + label_embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_fc_dim + label_embedding_dim, latent_dim)

        self.conv_decoder_initial_size = conv_decoder_initial_size
        self.conv_decoder_initial_channels = conv_decoder_initial_channels
        self.decoder_fc_in_features = conv_decoder_fc_in_features

        self.fc_dec = nn.Linear(latent_dim + label_embedding_dim, self.decoder_fc_in_features)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(self.conv_decoder_initial_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, y_idx):
        x_seq = x.squeeze(1)
        encoder_outputs, hidden = self.encoder_rnn(x_seq)
        
        if self.encoder_variant == "full_output_mean_pool":
            rnn_features = torch.mean(encoder_outputs, dim=1)
        elif self.encoder_variant == "last_hidden":
            if self.rnn_type == 'lstm': 
                h_n, _ = hidden
            else: 
                h_n = hidden
            if self.num_directions == 2:
                 rnn_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            else:
                 rnn_features = h_n[-1]

        y_embed = self.label_embedding(y_idx)
        combined_features = torch.cat([rnn_features, y_embed], dim=1)
        return self.fc_mu(combined_features), self.fc_logvar(combined_features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_idx):
        y_embed = self.label_embedding(y_idx)
        combined_latent = torch.cat([z, y_embed], dim=1)
        h = self.fc_dec(combined_latent)
        h = h.view(h.size(0), self.conv_decoder_initial_channels, 
                   self.conv_decoder_initial_size, self.conv_decoder_initial_size)
        return self.decoder_conv(h)

    def forward(self, x, y_idx):
        mu, logvar = self.encode(x, y_idx)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y_idx), mu, logvar, z


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
    
# --- Training Function ---
def train_conditional_vae(model, loader, num_epochs, learning_rate, beta=1.0, run_output_dir="."):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # history = {"train_loss": [], "train_recon": [], "train_kld": []} # Optional detailed history
    
    tqdm.write(f"\nTraining Conditional {model.__class__.__name__}...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        ep_loss, ep_recon, ep_kld = 0,0,0
        for x_batch, y_batch_idx in tqdm(loader, desc=f"Cond. Epoch {epoch}/{num_epochs}", leave=False):
            x_batch, y_batch_idx = x_batch.to(device), y_batch_idx.to(device)
            recon, mu, logvar, _ = model(x_batch, y_batch_idx)
            loss, recon_l, kld_l = vae_loss_fn(x_batch, recon, mu, logvar, beta)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss += loss.item() * x_batch.size(0)
            ep_recon += recon_l.item() * x_batch.size(0)
            ep_kld += kld_l.item() * x_batch.size(0) 


        final_epoch_loss = ep_loss / len(loader.dataset)
        final_epoch_recon = ep_recon / len(loader.dataset)
        final_epoch_kld = ep_kld / len(loader.dataset)
        print_msg = (f"Cond. Epoch {epoch:02d} | Train Loss: {final_epoch_loss:.4f} \n(R: {final_epoch_recon:.4f}, K: {final_epoch_kld:.4f})")
        tqdm.write(print_msg)

    torch.save(model.state_dict(), os.path.join(run_output_dir, "conditional_model_final_weights.pth"))
    tqdm.write(f"Conditional model training complete. Weights saved in {run_output_dir}")
    return model


if __name__ == "__main__":
    # --- HYPERPARAMETERS ---   
    EPOCHS = 500
    LEARNING_RATE = 5e-4
    BETA = 1.0
    BATCH_SIZE = 64
    label_embedding_dim = 16
    # CONV VAE SETTINGS
    # CHOSEN_CONDITIONAL_ARCH = "conv"
    # best_latent_dim = 64
    # conv_initial_size = 4
    # conv_initial_channels = 64
    ### RNN_Conv VAE SETTINGS
    CHOSEN_CONDITIONAL_ARCH = "rnn_conv"
    best_latent_dim = 64 
    rnn_type = 'gru'
    rnn_hidden_dim = 128
    rnn_num_layers = 3  
    rnn_encoder_variant = 'last_hidden'
    rnn_bidirectional = True
    conv_dec_init_channels = 64 
    conv_dec_init_size = 4
    conv_dec_fc_in = conv_dec_init_channels * conv_dec_init_size * conv_dec_init_size
    # ------

    set_seed(42)
    tqdm.write(f"Using device: {device}")

    target_class_name_to_id = {
        "rabbit": 0,  
        "yoga": 1,    
        "snowman": 3 
    }
   
    target_class_names = list(target_class_name_to_id.keys())
    target_class_ids_original = [target_class_name_to_id[name] for name in target_class_names]

    full_train_images_np = np.load("quickdraw_subset_np/train_images.npy")
    full_train_labels_np = np.load("quickdraw_subset_np/train_labels.npy")
 
    subset_indices = np.isin(full_train_labels_np, target_class_ids_original)

    subset_images_np = full_train_images_np[subset_indices]
    subset_labels_original_np = full_train_labels_np[subset_indices]

    label_map_original_to_new = {original_id: new_id for new_id, original_id in enumerate(target_class_ids_original)}
    subset_labels_mapped_np = np.array([label_map_original_to_new[l] for l in subset_labels_original_np]).astype(np.int64)
    num_conditional_classes = len(target_class_ids_original)

    subset_images_t = (torch.from_numpy(subset_images_np).float() / 255.0).unsqueeze(1)
    subset_labels_t = torch.from_numpy(subset_labels_mapped_np).long()
    
    conditional_dataset = TensorDataset(subset_images_t, subset_labels_t)
    conditional_loader = DataLoader(conditional_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    cond_run_name = f"Cond_{CHOSEN_CONDITIONAL_ARCH}_ld{best_latent_dim}"
    if CHOSEN_CONDITIONAL_ARCH == "rnn_conv":
        cond_run_name += f"_{rnn_type}"
    cond_run_output_dir = os.path.join(COND_OUTPUT_BASE_DIR, cond_run_name)
    os.makedirs(cond_run_output_dir, exist_ok=True)

    if CHOSEN_CONDITIONAL_ARCH == "conv":
        conditional_model = ConditionalConvVAE(
            latent_dim=best_latent_dim,
            num_classes=num_conditional_classes,
            label_embedding_dim=label_embedding_dim,
            initial_size=conv_initial_size,
            initial_channels_encoder=conv_initial_channels
        ).to(device)
    elif CHOSEN_CONDITIONAL_ARCH == "rnn_conv":
        conditional_model = ConditionalRNNEncoderConvDecoderVAE(
            rnn_type=rnn_type, 
            input_dim=28, 
            hidden_dim_rnn=rnn_hidden_dim,
            num_layers_rnn=rnn_num_layers,
            latent_dim=best_latent_dim,
            num_classes=num_conditional_classes, encoder_variant=rnn_encoder_variant,
            bidirectional_encoder=rnn_bidirectional,
            conv_decoder_fc_in_features=conv_dec_fc_in,
            conv_decoder_initial_channels=conv_dec_init_channels,
            conv_decoder_initial_size=conv_dec_init_size,
            label_embedding_dim=label_embedding_dim
        ).to(device)


    trained_conditional_model = train_conditional_vae(
        conditional_model, conditional_loader,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        run_output_dir=cond_run_output_dir
    )
    
    trained_conditional_model.eval()
    num_samples_per_category = 5
    

    classifier_model = MLP(input_size=28*28, hidden_size=128, output_size=5)
    classifier_model.load_model("classifier.pkl")
    classifier_model.to(device).eval()

    for new_label_idx in range(num_conditional_classes):
        class_name_for_generation = target_class_names[new_label_idx]

        y_indices_for_gen = torch.tensor([new_label_idx] * num_samples_per_category, device=device).long()
        z_for_gen = torch.randn(num_samples_per_category, trained_conditional_model.latent_dim).to(device)
        
        with torch.no_grad():
            generated_samples = trained_conditional_model.decode(z_for_gen, y_indices_for_gen)
        
        save_path = os.path.join(cond_run_output_dir, f"generated_samples_{class_name_for_generation.replace(' ','_')}.png")
        save_image_grid_conditional(generated_samples, save_path, n_rows=1)
    
        with torch.no_grad():
            generated_samples = generated_samples.view(num_samples_per_category, 1, -1)

            predictions = classifier_model(generated_samples.to(device)) 
            probabilities = F.softmax(predictions, dim=2)
            predicted_indices = torch.argmax(probabilities, dim=2)

            tqdm.write(f"\nClassifier predictions for generated '{class_name_for_generation}':")
            for i in range(num_samples_per_category):
                pred_idx = predicted_indices[i][0].item()
                predicted_class_name = "Unknown"
                if pred_idx in target_class_ids_original:
                        class_name_idx = target_class_ids_original.index(pred_idx)
                        predicted_class_name = target_class_names[class_name_idx]

                tqdm.write(f"  Sample {i+1}: Predicted as '{predicted_class_name}' (index {pred_idx}), Confidence: {probabilities[i,0,pred_idx].item():.3f}")


    tqdm.write(f"\nConditional VAE completed. Check '{cond_run_output_dir}' for outputs.")