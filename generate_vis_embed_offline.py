import numpy as np
import torch
import torch.nn as nn
from torch import optim
from numpy.random import choice
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import sys
from omegaconf import OmegaConf
from tempo.data_provider.data_factory import data_provider
from tempo.utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from torch.utils.data import Subset
from tqdm import tqdm
from tempo.models.PatchTST import PatchTST
from tempo.models.GPT4TS import GPT4TS
from tempo.models.DLinear import DLinear
from tempo.models.T5 import T54TS
from tempo.models.ETSformer import ETSformer
from tempo.models.Offline import TEMPO


def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

warnings.filterwarnings('ignore')

# Fix random seed for reproducibility
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def print_dataset_info(data, loader, name="Dataset"):
    print(f"\n=== {name} Information ===")
    print(f"Number of samples: {len(data)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches: {len(loader)}")
    for attr in ['features', 'targets', 'shape']:
        if hasattr(data, attr):
            print(f"{attr}: {getattr(data, attr)}")

def prepare_data_loaders(args, config):
    """
    Prepare train, validation and test data loaders.
    
    Args:
        args: Arguments containing dataset configurations
        config: Configuration dictionary
    
    Returns:
        tuple: (train_data, train_loader, test_data, test_loader, val_data, val_loader)
    """
    train_datas = []
    val_datas = []
    min_sample_num = sys.maxsize
    # First pass to get validation data and minimum sample number
    for dataset_name in args.datasets.split(','):
        _update_args_from_config(args, config, dataset_name)
        
        train_data, train_loader = data_provider(args, 'train')
        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange', 'monash']:
            min_sample_num = min(min_sample_num, len(train_data))
    
    for dataset_name in args.eval_data.split(','):  
        _update_args_from_config(args, config, dataset_name)  
        val_data, val_loader = data_provider(args, 'val') 
        val_datas.append(val_data)

    # Second pass to prepare training data with proper sampling
    for dataset_name in args.datasets.split(','):
        _update_args_from_config(args, config, dataset_name)
        
        train_data, _ = data_provider(args, 'train')
        
        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange', 'monash'] and args.equal == 1:
            train_data = Subset(train_data, choice(len(train_data), min_sample_num))
            
        if args.equal == 1:
            if dataset_name == 'electricity' and args.electri_multiplier > 1:
                train_data = Subset(train_data, choice(len(train_data), 
                                  int(min_sample_num * args.electri_multiplier)))
            elif dataset_name == 'traffic' and args.traffic_multiplier > 1:
                train_data = Subset(train_data, choice(len(train_data),
                                  int(min_sample_num * args.traffic_multiplier)))
                
        train_datas.append(train_data)

    # Combine datasets if multiple exist
    if len(train_datas) > 1:
        train_data = _combine_datasets(train_datas)
        val_data = _combine_datasets(val_datas)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    
    # Prepare test data
    _update_args_from_config(args, config, args.target_data)
    test_data, test_loader = data_provider(args, 'test')

    print_dataset_info(train_data, train_loader, "Training Dataset")
    print_dataset_info(val_data, val_loader, "Validation Dataset")
    print_dataset_info(test_data, test_loader, "Test Dataset")
    
    return train_data, train_loader, test_data, test_loader, val_data, val_loader

def _update_args_from_config(args, config, dataset_name):
    """Update args with dataset specific configurations"""
    dataset_config = config['datasets'][dataset_name]
    for key in ['data', 'root_path', 'data_path', 'data_name', 'features',
                'freq', 'target', 'embed', 'percent', 'lradj']:
        setattr(args, key, getattr(dataset_config, key))
    
    if args.freq == 0:
        args.freq = 'h'

def _combine_datasets(datasets):
    """Combine multiple datasets into one"""
    combined = datasets[0]
    for dataset in datasets[1:]:
        combined = torch.utils.data.ConcatDataset([combined, dataset])
    return combined

def compute_vision_embeddings(model, data_loader, device, save_dir, data, loader_type):
    """Compute embeddings using TEMPO's vision-based method."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        trend_list, season_list, noise_list = [] , [] , []
        for batch_x, _, _, _, _, _, _, _, _, _, in tqdm(data_loader, total = len(data_loader)):
            batch_x = batch_x.float().to(device)
            trend_embed, season_embed, noise_embed = model.compute_vision_embeddings(batch_x, save_dir)
            trend_list.append(trend_embed)
            season_list.append(season_embed)
            noise_list.append(noise_embed)
            
    trend_tensor = torch.stack(trend_list).unsqueeze(1)
    season_tensor = torch.stack(season_list).unsqueeze(1)
    noise_tensor = torch.stack(noise_list).unsqueeze(1)
    # print(trend_tensor.shape)
    data = data.lower()
    torch.save(trend_tensor, os.path.join(save_dir, f'{data}_trend_embedding_{loader_type}.pth'))
    # print(f'{data}_trend_embedding_{loader_type}.pth')
    torch.save(season_tensor, os.path.join(save_dir, f'{data}_season_embedding_{loader_type}.pth'))
    torch.save(noise_tensor, os.path.join(save_dir, f'{data}_noise_embedding_{loader_type}.pth'))
    print(f"Vision embeddings saved in {save_dir}")


# --- Main Execution ---
parser = argparse.ArgumentParser(description="Compute Vision Embeddings for ETTh1")
## General Arguments
parser.add_argument('--model_id', type=str, default='tempo_etth1_multi-debug') # Specifies a unique identifier for the model instance being trained
parser.add_argument('--checkpoints', type=str, default='checkpoints/') # Directory path where model checkpoints are saved
parser.add_argument('--task_name', type=str, default='long_term_forecast') # Defines the type of task, e.g., classification/anomaly_detection/ long_term_forecast/imputation/ short_term_forecasting
parser.add_argument('--prompt', type=int, default=0) # Indicates whether prompt-based training is used (0 = No, 1 = Yes).
parser.add_argument('--num_nodes', type=int, default=1) # Number of features but always set to 1 ????
## Sequence and Label Settings
parser.add_argument('--seq_len', type=int, default=512) # Length of the input sequence to the model.
parser.add_argument('--pred_len', type=int, default=96) # Number of steps to predict in the output sequence (horizon)
parser.add_argument('--label_len', type=int, default=48) #  Length of the label sequence used in the training phase to calculate the loss.
## Optimization Settings
parser.add_argument('--decay_fac', type=float, default=0.9) # Factor by which learning rate decays during training.
parser.add_argument('--learning_rate', type=float, default=0.001) # Initial learning rate for training.
parser.add_argument('--batch_size', type=int, default=128) # Batch size for training.
parser.add_argument('--num_workers', type=int, default=0) # Number of worker threads for data loading.
parser.add_argument('--train_epochs', type=int, default=10) # Number of training epochs.
parser.add_argument('--lradj', type=str, default='type3') # Learning rate adjustment strategy
parser.add_argument('--patience', type=int, default=5)
## Model and Architecture Settings
parser.add_argument('--gpt_layers', type=int, default=6) # Number of layers in the GPT
parser.add_argument('--is_gpt', type=int, default=1) # Flag to indicate whether GPT-style modeling is used (1 = Yes, 0 = No).
parser.add_argument('--e_layers', type=int, default=3) # Number of encoder layers in the model.
parser.add_argument('--d_model', type=int, default=768) # Dimensionality of model embeddings.
parser.add_argument('--n_heads', type=int, default=4) # Number of (attention?) heads in the model.
parser.add_argument('--d_ff', type=int, default=768) # Dimensionality of the feed-forward network in the transformer layers.
parser.add_argument('--dropout', type=float, default=0.3) # Dropout rate used in the model to prevent overfitting.
parser.add_argument('--enc_in', type=int, default=7) # Number of input features for the encoder.
parser.add_argument('--c_out', type=int, default=7) # Number of output channels/features.
parser.add_argument('--patch_size', type=int, default=16) # Size of the patches for patch-based learning.
parser.add_argument('--kernel_size', type=int, default=25) # In TEMPO - define the size of the window over which the moving average is computed
## Loss and Training Strategy
parser.add_argument('--loss_func', type=str, default='mse') # Loss function used during training
parser.add_argument('--pretrain', type=int, default=1) # Flag to indicate whether to load or not the pre-trained model(GPT2) (1 = Yes, 0 = No).
parser.add_argument('--freeze', type=int, default=1) # Flag to freeze specific model components during fine-tuning (1 = Yes, 0 = No).
parser.add_argument('--model', type=str, default='TEMPO') 
parser.add_argument('--stride', type=int, default=8) # Specifies the number of steps the window moves after each operation
parser.add_argument('--max_len', type=int, default=-1) # Maximum sequence length; -1 indicates no limit.
parser.add_argument('--hid_dim', type=int, default=16) # Hidden dimension size for specific layers.
parser.add_argument('--tmax', type=int, default=10) # The number of epochs or steps over which the learning rate will decrease from its initial value to the minimum (eta_min)
parser.add_argument('--itr', type=int, default=3) # Number of iterations for a particular forward pass of the model
parser.add_argument('--cos', type=int, default=0) #  When cos is set to 1, cosine annealing will control the learning rate decay
parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: dont do the equal sampling')
parser.add_argument('--pool', action='store_true', help='whether use prompt pool')
parser.add_argument('--no_stl_loss', action='store_true', help='whether use Season-Trend Loss')
# Dataset and Configurations
parser.add_argument('--stl_weight', type=float, default=0.01) # Weight of STL (Season-Trend Loss) in the total loss computation
parser.add_argument('--config_path', type=str, default='./configs/multiple_datasets.yml') # Path to the configuration YAML file for multiple datasets
parser.add_argument('--datasets', type=str, default='ETTh1') # Specifies the datasets used for training
parser.add_argument('--target_data', type=str, default='ETTh1') # Indicates the target dataset for forecasting.
#eval_data
parser.add_argument('--eval_data', type=str, default='ETTh1') # Dataset used for evaluation.
parser.add_argument('--use_token', type=int, default=0) # If use prompt token's representation as the forecasting's information
parser.add_argument('--electri_multiplier', type=int, default=1) # Multiplier for electric data scaling.
parser.add_argument('--traffic_multiplier', type=int, default=1) # Multiplier for traffic data scaling.
parser.add_argument('--embed', type=str, default='timeF') # Type of embedding used (e.g., timeF for time-frequency embeddings).
parser.add_argument('--vision', type=int, default=0) # Flag to indicate whether vision-based models are used (1 = Yes, 0 = No).
parser.add_argument('--vis_encoder_dim', type=int, default=512) # Dimensionality of the vision encoder.
parser.add_argument("--save_dir", type=str, default="/home/arielsi/VisionaryTimes/Pics_embed")
parser.add_argument('--create_offline_vision', type=int, default=1) 


args = parser.parse_args()
config = get_init_config(args.config_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TEMPO(args, device).to(device)

# Prepare data loaders (only need the test loader for computing embeddings)
train_data, train_loader, test_data, test_loader, val_data, val_loader = prepare_data_loaders(args, config)

# train_data, train_loader, test_data, test_loader = prepare_data_loaders(args, config)[0:4]

# Compute vision embeddings for train
compute_vision_embeddings(model, train_loader, device, args.save_dir, args.target_data, "train")

# Compute vision embeddings for validation
compute_vision_embeddings(model, val_loader, device, args.save_dir, args.target_data, "val")

# Compute vision embeddings for test
compute_vision_embeddings(model, test_loader, device, args.save_dir, args.target_data, "test")