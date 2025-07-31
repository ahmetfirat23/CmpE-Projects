# unconditional_vaes.py
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
import torchmetrics
from torchvision.utils import make_grid, save_image
import pathlib

# --- Utility Functions ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_BASE_DIR = "outputs_unconditional"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def save_loss_plot(history, title, filepath):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="Train Total")
    plt.plot(history["val_loss"], label="Val Total")
    plt.title("Total VAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history["train_recon"], label="Train Recon")
    plt.plot(history["val_recon"], label="Val Recon")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history["train_kld"], label="Train KLD")
    plt.plot(history["val_kld"], label="Val KLD")
    plt.title("KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history["val_recon"],label="Val Recon")
    plt.plot(history["val_kld"],label="Val KLD")
    plt.title("Val: Reconstruction Loss and KLD")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(filepath)
    plt.close()

def save_image_grid(images_tensor, filepath, n_rows=4, normalize=True, scale_each=True):
    if images_tensor.ndim==3:
        images_tensor=images_tensor.unsqueeze(1)
        
    actual_n_rows = min(n_rows,images_tensor.size(0)) if images_tensor.size(0)>0 else 1

    if actual_n_rows > 0:
        grid=make_grid(images_tensor.cpu(),nrow=actual_n_rows,normalize=normalize,scale_each=scale_each,pad_value=0.5)
        save_image(grid,filepath)
        print(f"Saved image grid: {filepath}")

def prepare_images_for_metric(x,target_size=(299,299)):
    if x.ndim==3:x=x.unsqueeze(1)
    if x.size(1)==1:x=x.repeat(1,3,1,1)
    if x.size(2)!=target_size[0] or x.size(3)!=target_size[1]:
        x=F.interpolate(x,size=target_size,mode="bilinear",align_corners=False)
    return x

# --- Model Definitions ---
class RNNEncoderConvDecoderVAE(nn.Module): 
    # conv_decoder_fc_in_features == conv_decoder_initial_channels * conv_decoder_initial_size * conv_decoder_initial_size
    def __init__(self, 
                 rnn_type='lstm', 
                 input_dim=28, 
                 hidden_dim_rnn=128, 
                 num_layers_rnn=2, 
                 latent_dim=32, 
                 encoder_variant="last_hidden", 
                 bidirectional_encoder=False, 
                 conv_decoder_fc_in_features=1024,   conv_decoder_initial_channels=64, 
                 conv_decoder_initial_size=4):
        super().__init__()
        self.rnn_type=rnn_type.lower()
        self.seq_len=input_dim
        self.input_dim_rnn=input_dim
        self.latent_dim=latent_dim
        self.hidden_dim_rnn=hidden_dim_rnn
        self.num_layers_rnn=num_layers_rnn
        self.encoder_variant=encoder_variant
        self.num_directions=2 if bidirectional_encoder else 1
        rnn_module=nn.LSTM if self.rnn_type=='lstm' else nn.GRU
        encoder_output_fc_dim=hidden_dim_rnn*self.num_directions
        self.conv_decoder_initial_size=conv_decoder_initial_size
        self.conv_decoder_initial_channels=conv_decoder_initial_channels
        self.decoder_fc_in_features=conv_decoder_fc_in_features 

        self.encoder_rnn=rnn_module(self.input_dim_rnn, hidden_dim_rnn,num_layers_rnn, batch_first=True, bidirectional=bidirectional_encoder)  
        self.fc_mu=nn.Linear(encoder_output_fc_dim,latent_dim)
        self.fc_logvar=nn.Linear(encoder_output_fc_dim,latent_dim)
        self.fc_dec=nn.Linear(latent_dim,self.decoder_fc_in_features)
        self.decoder_conv=nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.conv_decoder_initial_channels,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=1,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid())
        
    def encode(self,x): 
        x_seq=x.squeeze(1)
        encoder_outputs, hidden=self.encoder_rnn(x_seq)
        if self.encoder_variant=="full_output_mean_pool":
            processed_output=torch.mean(encoder_outputs,dim=1)
        elif self.encoder_variant=="last_hidden":
            if self.rnn_type=='lstm':
                h_n,_=hidden
            else:
                h_n=hidden
            if self.num_directions==2:
                processed_output=torch.cat((h_n[-2,:,:],h_n[-1,:,:]),dim=1)
            else:
                processed_output=h_n[-1,:,:]
        return self.fc_mu(processed_output), self.fc_logvar(processed_output)
    
    def reparameterize(self,z_mu,z_logvar):
        std=torch.exp(0.5*z_logvar)
        eps=torch.randn_like(std)
        return z_mu+eps*std
    
    def decode(self,z):
        h=self.fc_dec(z)
        h=h.view(h.size(0),self.conv_decoder_initial_channels,self.conv_decoder_initial_size,self.conv_decoder_initial_size)
        return self.decoder_conv(h)
    
    def forward(self,x):
        z_mu,z_logvar=self.encode(x)
        z=self.reparameterize(z_mu,z_logvar)
        return self.decode(z),z_mu,z_logvar,z

class ConvVAE(nn.Module): 
    def __init__(self,
                 latent_dim=32,
                 input_channels=1,
                 initial_size=4,
                 initial_channels_encoder=64):
        super().__init__()
        self.latent_dim=latent_dim
        self.initial_size=initial_size
        self.initial_channels_encoder=initial_channels_encoder
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=16,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(True), 
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(True),             
            nn.Conv2d(in_channels=32,
                      out_channels=self.initial_channels_encoder,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(True))
        self.fc_in_features=self.initial_channels_encoder*self.initial_size*self.initial_size
        self.fc_mu=nn.Linear(self.fc_in_features,latent_dim)
        self.fc_logvar=nn.Linear(self.fc_in_features,latent_dim)
        self.fc_dec=nn.Linear(latent_dim,self.fc_in_features)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.initial_channels_encoder,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=0),
            nn.ReLU(True), 
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(True),          
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=input_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid())
        
    def encode(self,x):
        h=self.encoder(x)
        h=h.view(h.size(0),-1)
        return self.fc_mu(h),self.fc_logvar(h)
    
    def reparameterize(self,z_mu,z_logvar):
        std=torch.exp(0.5*z_logvar)
        eps=torch.randn_like(std)
        return z_mu+eps*std
    
    def decode(self,z):
        h=self.fc_dec(z)
        h=h.view(h.size(0),self.initial_channels_encoder,self.initial_size,self.initial_size)
        return self.decoder(h)
    
    def forward(self,x):
        z_mu,z_logvar=self.encode(x)
        z=self.reparameterize(z_mu,z_logvar)
        return self.decode(z),z_mu,z_logvar,z

# --- VAE Loss Function ---
def vae_loss_fn(x, x_recon, z_mu, z_logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1).mean()
    return recon_loss + beta * kld, recon_loss, kld

# --- Training Function ---
def train_vae(model_config, train_ds, val_ds, test_ds, run_output_dir):
    model_arch = model_config['model_arch']
    lr = model_config['lr']
    epochs = model_config['epochs']
    batch_size = model_config['batch_size']
    beta = model_config.get('beta', 1.0)
    latent_dim = model_config['latent_dim']

    if model_arch == 'rnn_conv':
        model = RNNEncoderConvDecoderVAE(
            rnn_type=model_config['rnn_type'], 
            hidden_dim_rnn=model_config['hidden_dim'],
            num_layers_rnn=model_config['num_layers'], 
            latent_dim=latent_dim,
            encoder_variant=model_config['encoder_variant'],
            bidirectional_encoder=model_config.get('bidirectional', False)).to(device)
    elif model_arch == 'conv_conv':
        model = ConvVAE(latent_dim=latent_dim).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kld": [], "val_kld": []}
    
    best_val_loss_for_this_run = float('inf')
    best_model_state_dict = None 
    weights_path_for_this_run = os.path.join(run_output_dir, "best_model_weights.pth")

    tqdm.write(f"Starting training for: {run_output_dir.split(os.sep)[-1]} | Config: {model_config}")


    epoch_pbar_desc = f"{run_output_dir.split(os.sep)[-2][:15]}:{run_output_dir.split(os.sep)[-1][:20]}" 
    epoch_pbar = tqdm(range(1, epochs + 1), desc=epoch_pbar_desc, unit="ep", leave=False)

    for epoch in epoch_pbar:
        model.train()
        ep_train_loss,ep_train_recon,ep_train_kld = 0.0,0.0,0.0

        for imgs,_ in train_loader:
            imgs=imgs.to(device)
            optimizer.zero_grad()
            recon,z_mu,z_logvar,_=model(imgs)
            loss,recon_l,kld_l=vae_loss_fn(imgs,recon,z_mu,z_logvar,beta)
            loss.backward()
            optimizer.step()
            ep_train_loss+=loss.item()*imgs.size(0)
            ep_train_recon+=recon_l.item()*imgs.size(0)
            ep_train_kld+=kld_l.item()*imgs.size(0)

        history["train_loss"].append(ep_train_loss/len(train_loader.dataset))
        history["train_recon"].append(ep_train_recon/len(train_loader.dataset))
        history["train_kld"].append(ep_train_kld/len(train_loader.dataset))
        
        model.eval()
        ep_val_loss,ep_val_recon,ep_val_kld = 0.0,0.0,0.0
        with torch.no_grad():
            for imgs,_ in val_loader:
                imgs=imgs.to(device)
                recon,z_mu,z_logvar,_=model(imgs)
                loss,recon_l,kld_l=vae_loss_fn(imgs,recon,z_mu,z_logvar,beta)
                ep_val_loss+=loss.item()*imgs.size(0)
                ep_val_recon+=recon_l.item()*imgs.size(0)
                ep_val_kld+=kld_l.item()*imgs.size(0)

        current_val_loss=ep_val_loss/len(val_loader.dataset)
        history["val_loss"].append(current_val_loss)
        history["val_recon"].append(ep_val_recon/len(val_loader.dataset))
        history["val_kld"].append(ep_val_kld/len(val_loader.dataset))
        
        status_indicator = "" 
        if current_val_loss < best_val_loss_for_this_run:
            best_val_loss_for_this_run = current_val_loss
            best_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            status_indicator = "*"
            
        postfix_dict = {
            'TrL': history['train_loss'][-1], 
            'VL': current_val_loss,
            'BestVL': best_val_loss_for_this_run,
        }
        if status_indicator:
             postfix_dict['Status'] = status_indicator
            
        epoch_pbar.set_postfix(postfix_dict, refresh=True) 
    epoch_pbar.close() 

    torch.save(best_model_state_dict, weights_path_for_this_run)
    tqdm.write(f"Best weights for {run_output_dir.split(os.sep)[-1]} saved. ValLoss: {best_val_loss_for_this_run:.4f}")
    model.load_state_dict(best_model_state_dict) 

    plot_path = os.path.join(run_output_dir,"loss_plot.png")
    save_loss_plot(history,f"Loss-{model_config['model_arch']}_{model_config.get('rnn_type','na')}_LD{latent_dim}", plot_path)
    
    if os.path.exists(weights_path_for_this_run): 
        model.load_state_dict(torch.load(weights_path_for_this_run, map_location=device))
    elif best_model_state_dict: 
         model.load_state_dict(best_model_state_dict)    

    generate_and_save_reconstructions(model=model,
                                      dataloader=train_loader,
                                      num_pairs=8,
                                      filepath=os.path.join(run_output_dir,"recons_train.png"))
    generate_and_save_reconstructions(model=model,
                                      dataloader=train_loader,
                                      num_pairs=8,
                                      filepath=os.path.join(run_output_dir,"recons_val.png"))
    generate_and_save_reconstructions(model=model,
                                      dataloader=train_loader,
                                      num_pairs=8,
                                      filepath=os.path.join(run_output_dir,"recons_test.png"))
    
    return model, history, best_val_loss_for_this_run, weights_path_for_this_run

# --- Sample Generation and Reconstruction Saving ---
@torch.no_grad()
def generate_and_save_samples(model, num_samples, filepath):
    model.eval()
    latent_dim = model.latent_dim
    z = torch.randn(num_samples, latent_dim).to(device)
    samples = model.decode(z)
    save_image_grid(samples, filepath, n_rows=int(np.sqrt(num_samples)))


@torch.no_grad()
def generate_and_save_reconstructions(model, dataloader, num_pairs, filepath):
    model.eval()
    originals_batch,_=next(iter(dataloader))
    originals=originals_batch[:num_pairs].to(device)
    reconstructions,_,_,_=model(originals) 
    comparison=torch.cat([originals.cpu(),reconstructions.cpu()])
    save_image_grid(comparison,filepath,n_rows=2)

# --- Main Execution & Grid Search ---
if __name__ == "__main__":
    set_seed(42)
    tqdm.write(f"Using device: {device}")

    train_images_np=np.load("quickdraw_subset_np/train_images.npy")
    train_labels_np=np.load("quickdraw_subset_np/train_labels.npy")
    test_images_np=np.load("quickdraw_subset_np/test_images.npy")
    test_labels_np=np.load("quickdraw_subset_np/test_labels.npy")

    train_images_t=(torch.from_numpy(train_images_np).float()/255.0).unsqueeze(1)
    train_labels_t=torch.from_numpy(train_labels_np).long()
    test_images_t=(torch.from_numpy(test_images_np).float()/255.0).unsqueeze(1)
    test_labels_t=torch.from_numpy(test_labels_np).long()

    val_split_idx=int(0.9*len(train_images_t))
    train_ds=TensorDataset(train_images_t[:val_split_idx],train_labels_t[:val_split_idx])
    val_ds=TensorDataset(train_images_t[val_split_idx:],train_labels_t[val_split_idx:])
    test_ds=TensorDataset(test_images_t,test_labels_t)

    N_FID_REAL_SAMPLES=256
    fid_real_images_subset=train_ds.tensors[0][:N_FID_REAL_SAMPLES].to(device)
    fid_real_images_for_metric_global=prepare_images_for_metric(fid_real_images_subset)


    EPOCHS = 1
    BATCH_SIZE = 64
    rnn_conv_param_grid=[]
    baseline_rnn_structure={'hidden_dim':128,
                            'num_layers':1,
                            'encoder_variant':'last_hidden','bidirectional':False}
    for rnn_type_val in ['lstm','gru']:
        for ld_val in [32,64]: 
            for beta_val in [0.5,1.0,2.0]: 
                for lr_val in [1e-3,5e-4]:
                    config={'model_arch':'rnn_conv',
                            'rnn_type':rnn_type_val,
                            'latent_dim':ld_val,
                            **baseline_rnn_structure,
                            'lr':lr_val,
                            'epochs':EPOCHS,
                            'batch_size':BATCH_SIZE,
                            'beta':beta_val}
                    rnn_conv_param_grid.append(config)

    structural_exploration_defaults={'latent_dim':64,
                                     'beta':1.0,
                                     'lr':1e-3}
    for rnn_type_val_struct in ['lstm','gru']:
        for hd_val in [64,128]: 
            for nl_val in [1,2]:
                for enc_var_val in ['last_hidden','full_output_mean_pool']:
                    for bi_val in [False,True]:
                        config={'model_arch':'rnn_conv',
                                'rnn_type':rnn_type_val_struct,
                                **structural_exploration_defaults,
                                'hidden_dim':hd_val,
                                'num_layers':nl_val,'encoder_variant':enc_var_val,'bidirectional':bi_val,
                                'epochs':EPOCHS,
                                'batch_size':BATCH_SIZE}
                        rnn_conv_param_grid.append(config)

    cnn_param_grid=[]
    for ld_val in [32,64,128]:
        for lr_val in [1e-3,5e-4]:
            for beta_val in [0.5,1.0,2.0,4.0]:
                config={'model_arch':'conv_conv',
                        'latent_dim':ld_val,
                        'lr':lr_val,
                        'epochs':EPOCHS,
                        'batch_size':BATCH_SIZE,
                        'beta':beta_val}
                cnn_param_grid.append(config)

    all_param_grids_with_names = {
        "RNN_Encoder_Conv_Decoder_VAE":rnn_conv_param_grid,
        "Conv_VAE":cnn_param_grid
        }
    overall_best_models_summary={}


    total_configurations_all_families = sum(len(grid) for grid in all_param_grids_with_names.values())
    overall_grid_pbar = tqdm(total=total_configurations_all_families, desc="Grid Search", position=0, leave=True)

    for grid_family_name, param_grid in all_param_grids_with_names.items():
        tqdm.write(f"\n\n--- Processing Model Family: {grid_family_name} ---")
        best_val_loss_for_family=float('inf')
        best_model_final_config_for_family=None
        best_model_weights_path_for_family=None
        best_model_output_dir_for_family=None

        for i, config in enumerate(param_grid):
            run_name_parts=[config['model_arch'],
                            config.get('rnn_type','na'),
                            f"ld{config['latent_dim']}",
                            f"b{config['beta']}",
                            f"lr{config['lr']}",
                            f"run{i+1}"]
            run_name="_".join(part for part in run_name_parts if part!='na')
            overall_grid_pbar.set_description(f"GS (Fam: {grid_family_name[:10]}.., Cfg: {run_name[:15]}..)")
            
            grid_family_fs_name=grid_family_name.replace(" ","_").replace("(","_").replace(")","_").replace(",","")
            run_output_dir=os.path.join(OUTPUT_BASE_DIR,grid_family_fs_name,run_name)
            os.makedirs(run_output_dir,exist_ok=True)
            
            _,_,current_run_best_val_loss,current_run_weights_path=train_vae(
                config,train_ds,val_ds,test_ds,run_output_dir)
            
            current_run_summary={'config':config,
                                 'best_val_loss':current_run_best_val_loss,
                                 'weights_path':current_run_weights_path,
                                 'output_dir':run_output_dir}
            with open(os.path.join(run_output_dir,"run_summary_val_loss_only.json"),'w') as f:
                json.dump(current_run_summary,f,indent=4)

            if current_run_best_val_loss<best_val_loss_for_family:
                best_val_loss_for_family=current_run_best_val_loss
                best_model_final_config_for_family=config 
                best_model_weights_path_for_family=current_run_weights_path
                best_model_output_dir_for_family=run_output_dir
            
            overall_grid_pbar.update(1)
        
        tqdm.write(f"\n--- Evaluating Best Model for Family: {grid_family_name} ---")
        if best_model_final_config_for_family and best_model_weights_path_for_family:
            tqdm.write(f"Best config for {grid_family_name}: {best_model_final_config_for_family}") # Log to main console via tqdm.write
            tqdm.write(f"Loading weights: {best_model_weights_path_for_family}")
            best_model_arch=best_model_final_config_for_family['model_arch'];best_latent_dim=best_model_final_config_for_family['latent_dim']
            if best_model_arch=='rnn_conv':
                final_best_model_for_family=RNNEncoderConvDecoderVAE(
                    rnn_type=best_model_final_config_for_family['rnn_type'],hidden_dim_rnn=best_model_final_config_for_family['hidden_dim'],
                    num_layers_rnn=best_model_final_config_for_family['num_layers'],latent_dim=best_latent_dim,
                    encoder_variant=best_model_final_config_for_family['encoder_variant'],
                    bidirectional_encoder=best_model_final_config_for_family.get('bidirectional',False)).to(device)
            elif best_model_arch=='conv_conv':
                final_best_model_for_family=ConvVAE(latent_dim=best_latent_dim).to(device)

            final_best_model_for_family.load_state_dict(torch.load(best_model_weights_path_for_family,map_location=device))
            num_metric_samples=min(128,N_FID_REAL_SAMPLES if N_FID_REAL_SAMPLES and N_FID_REAL_SAMPLES>0 else 128)
            samples_filepath=os.path.join(best_model_output_dir_for_family,"BEST_MODEL_generated_samples.png")
            generate_and_save_samples(final_best_model_for_family,num_metric_samples,samples_filepath, log_func=tqdm.write)
            is_mean,is_std,fid_score=-1.0,-1.0,-1.0 
            if fid_real_images_for_metric_global is not None and num_metric_samples > 0:
                final_best_model_for_family.eval()
                with torch.no_grad():
                    z_metric=torch.randn(num_metric_samples,final_best_model_for_family.latent_dim).to(device)
                    samples_for_metric=final_best_model_for_family.decode(z_metric)
                gen_imgs_for_metric=prepare_images_for_metric(samples_for_metric.cpu())
                is_metric_calc=torchmetrics.image.inception.InceptionScore(normalize=True).to(device)
                fid_metric_calc=torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(device)
                is_metric_calc.update(gen_imgs_for_metric.to(device));is_mean_t,is_std_t=is_metric_calc.compute();is_mean,is_std=is_mean_t.item(),is_std_t.item()
                fid_metric_calc.update(fid_real_images_for_metric_global.to(device),real=True);fid_metric_calc.update(gen_imgs_for_metric.to(device),real=False)
                fid_score=fid_metric_calc.compute().item()
                tqdm.write(f"Best Model {grid_family_name} - IS: {is_mean:.3f} ± {is_std:.3f}, FID: {fid_score:.3f}")
            else: tqdm.write(f"Skipping IS/FID for best {grid_family_name} model.")
            summary_for_reporting={'config':best_model_final_config_for_family,'best_val_loss':best_val_loss_for_family,
                                   'weights_path':best_model_weights_path_for_family,'output_dir':best_model_output_dir_for_family,
                                   'is_mean':is_mean,'is_std':is_std,'fid_score':fid_score}
            overall_best_models_summary[grid_family_name]=summary_for_reporting
            summary_file_path=os.path.join(best_model_output_dir_for_family,"BEST_MODEL_summary_for_reporting.json")
            with open(summary_file_path,'w') as f:
                json.dump(summary_for_reporting,f,default=lambda o:str(o) if isinstance(o,pathlib.Path)else'<not serializable>',indent=4)
            tqdm.write(f"Comprehensive summary for best {grid_family_name} model saved to: {summary_file_path}")
        else: tqdm.write(f"No successful runs for family: {grid_family_name}")

    overall_grid_pbar.close() # Close the main overall progress bar

    tqdm.write("\n\n--- Overall Grid Search Summary (Champions from each family) ---")
    tqdm.write("Refer to 'BEST_MODEL_summary_for_reporting.json' in each champion's output directory for detailed scores for your paper.")
    for family_name,summary in overall_best_models_summary.items():
        tqdm.write(f"\nChampion for {family_name}:")
        tqdm.write(f"  Output Dir: {summary['output_dir']}")
        tqdm.write(f"  Metrics: Val Loss={summary['best_val_loss']:.4f}, IS={summary['is_mean']:.3f} (±{summary['is_std']:.3f}), FID={summary['fid_score']:.3f}")
        tqdm.write(f"  Full details in: {os.path.join(summary['output_dir'],'BEST_MODEL_summary_for_reporting.json')}")
    tqdm.write(f"\nReminder for HW4: Use params from these best models for 'conditional_vaes.py'.")
    tqdm.write(f"All outputs saved in: '{OUTPUT_BASE_DIR}'")