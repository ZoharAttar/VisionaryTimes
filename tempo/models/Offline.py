import numpy as np
import torch
import torch.nn as nn
from torch import optim
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from tempo.embed import DataEmbedding, DataEmbedding_wo_time
from torchvision import models, transforms
from transformers import AutoProcessor, AutoModel
import torchvision.transforms as T
from transformers import SiglipModel, SiglipProcessor, SiglipVisionModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tempo.utils.rev_in import RevIn
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from huggingface_hub import hf_hub_download
import os
import warnings
from omegaconf import OmegaConf
import torch.nn.functional as F
import clip
import timm  
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.tsa.seasonal import STL


criterion = nn.MSELoss()
folder_id = "/Pics_embed"


class ComplexLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.fc_real = nn.Linear(input_dim, output_dim)
        self.fc_imag = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_real = torch.real(x)
        x_imag = torch.imag(x)
        out_real = self.fc_real(x_real) - self.fc_imag(x_imag)
        out_imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return torch.complex(out_real, out_imag)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        # f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        f"trainable params: {trainable_params} || all params: {all_param}"
    )

class MultiFourier(torch.nn.Module):
    def __init__(self, N, P):
        super(MultiFourier, self).__init__()
        self.N = N
        self.P = P
        self.a = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)
    
    def forward(self, t):
        output = torch.zeros_like(t)
        t = t.unsqueeze(-1).repeat(1, 1, max(self.N))  # shape: [batch_size, seq_len, max(N)]
        n = torch.arange(max(self.N)).unsqueeze(0).unsqueeze(0).to(t.device)  # shape: [1, 1, max(N)]
        for j in range(len(self.N)):  # loop over seasonal components
            # import ipdb; ipdb.set_trace() 
            cos_terms = torch.cos(2 * np.pi * (n[..., :self.N[j]]+1) * t[..., :self.N[j]] / self.P[j])  # shape: [batch_size, seq_len, N[j]]
            sin_terms = torch.sin(2 * np.pi * (n[..., :self.N[j]]+1) * t[..., :self.N[j]] / self.P[j])  # shape: [batch_size, seq_len, N[j]]
            output += torch.matmul(cos_terms, self.a[:self.N[j], j]) + torch.matmul(sin_terms, self.b[:self.N[j], j])
        return output

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class TEMPO(nn.Module):
    
    def __init__(self, configs, device):
        super(TEMPO, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.mul_season = MultiFourier([2], [24*4]) #, [24, 24*4])
        self.seq_len = configs.seq_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        # self.mlp = configs.mlp
        self.device = device
        self.vision = configs.vision
        self.use_components = configs.use_components
        self.vis_encoder_name = configs.vis_encoder_name

        ############--adding vision support--#################    
        if self.vision:
            if self.vis_encoder_name == "CLIP":
                self.vis_encoder_dim = 512
                self.vision_encoder, self.vision_encoder_preprocess = clip.load("ViT-B/32", device=self.device)
                self.vis_layer_trend = nn.Linear(configs.vis_encoder_dim, configs.d_model)
                self.vis_layer_season = nn.Linear(configs.vis_encoder_dim, configs.d_model)
                self.vis_layer_noise = nn.Linear(configs.vis_encoder_dim, configs.d_model)

            elif self.vis_encoder_name == "ViT":
                self.vis_encoder_dim = 384
                self.vision_encoder = timm.create_model('vit_small_patch16_224', pretrained=True)  # Loads pretrained ViT (small) with 16x16 patch size.
                self.vision_encoder.reset_classifier(0) # Removes the classification head to get raw image features.
                self.vision_encoder_preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])]) # Standard preprocessing for ViT models from timm (RGB normalized to [-1, 1]).
                self.vis_layer_trend = nn.Linear(configs.vis_encoder_dim, configs.d_model)
                self.vis_layer_season = nn.Linear(configs.vis_encoder_dim, configs.d_model)
                self.vis_layer_noise = nn.Linear(configs.vis_encoder_dim, configs.d_model)

            elif self.vis_encoder_name == "DeiT-Tiny":  # Data-efficient Image Transformer (DeiT) - Tiny variant
                self.vis_encoder_dim = 192
                self.vision_encoder = timm.create_model('deit_tiny_patch16_224', pretrained=True)
                self.vision_encoder.reset_classifier(0)
                self.vision_encoder_preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])  # Matches DeiT training normalization
                ])
                self.vis_layer_trend = nn.Linear(configs.vis_encoder_dim, configs.d_model)
                self.vis_layer_season = nn.Linear(configs.vis_encoder_dim, configs.d_model)
                self.vis_layer_noise = nn.Linear(configs.vis_encoder_dim, configs.d_model)
        ############--adding vision support--#################    

        self.map_trend = nn.Linear(configs.seq_len, configs.seq_len)
        self.map_season  = nn.Sequential(
            nn.Linear(configs.seq_len, 4*configs.seq_len),
            nn.ReLU(),
            nn.Linear(4*configs.seq_len, configs.seq_len)
        )

        # #self.map_season = nn.Linear(configs.seq_len, configs.seq_len)
        self.map_resid = nn.Linear(configs.seq_len, configs.seq_len)

        kernel_size = 25
        self.moving_avg = moving_avg(kernel_size, stride=1)

        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2_trend = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model  
                # self.gpt2_season = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                # self.gpt2_noise = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------No need to load pretrained GPT model------------------")
                self.gpt2_trend = GPT2Model(GPT2Config())
                # self.gpt2_season = GPT2Model(GPT2Config())
                # self.gpt2_noise = GPT2Model(GPT2Config())
            self.gpt2_trend.h = self.gpt2_trend.h[:configs.gpt_layers]
           
            self.prompt = configs.prompt
            
            # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
            self.gpt2_trend_token = self.tokenizer(text="Predict the future time step given the trend", return_tensors="pt").to(device)
            self.gpt2_season_token = self.tokenizer(text="Predict the future time step given the season", return_tensors="pt").to(device)
            self.gpt2_residual_token = self.tokenizer(text="Predict the future time step given the residual", return_tensors="pt").to(device)

            self.token_len = len(self.gpt2_trend_token['input_ids'][0])

            try:
                self.pool = configs.pool
                if self.pool:
                    self.prompt_record_plot = {}
                    self.prompt_record_id = 0
                    self.diversify = True

            except:
                self.pool = False

            if self.pool:
                self.prompt_key_dict = nn.ParameterDict({})
                self.prompt_value_dict = nn.ParameterDict({})
                # self.summary_map = nn.Linear(self.token_len, 1)
                self.summary_map = nn.Linear(self.patch_num, 1)
                self.pool_size = 30
                self.top_k = 3
                self.prompt_len = 3
                self.token_len = self.prompt_len * self.top_k
                for i in range(self.pool_size):
                    prompt_shape = (self.prompt_len, 768)
                    key_shape = (768)
                    self.prompt_value_dict[f"prompt_value_{i}"] = nn.Parameter(torch.randn(prompt_shape))
                    self.prompt_key_dict[f"prompt_key_{i}"] = nn.Parameter(torch.randn(key_shape))
            
                self.prompt_record = {f"id_{i}": 0 for i in range(self.pool_size)}
                self.prompt_record_trend = {}
                self.prompt_record_season = {}
                self.prompt_record_residual = {}
                self.diversify = True

        # projection layer which transform from patch size to d (d is model size)
        self.in_layer_trend = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer_season = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer_noise = nn.Linear(configs.patch_size, configs.d_model)
        # self.out_layer_noise = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.prompt == 1:
            # print((configs.d_model+9) * self.patch_num)
            self.use_token = configs.use_token
            if self.use_token == 1: # if use prompt token's representation as the forecasting's information
                    self.out_layer_trend = nn.Linear(configs.d_model * (self.patch_num+self.token_len), configs.pred_len)
                    self.out_layer_season = nn.Linear(configs.d_model * (self.patch_num+self.token_len), configs.pred_len)
                    self.out_layer_noise = nn.Linear(configs.d_model * (self.patch_num+self.token_len), configs.pred_len)
            else:
                self.out_layer_trend = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
                self.out_layer_season = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
                self.out_layer_noise = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
                # self.fre_len = configs.seq_len # // 2 + 1
                # self.out_layer_noise_fre = ComplexLinear(self.fre_len, configs.pred_len)
                # self.pred_len = configs.pred_len
                # self.seq_len = configs.seq_len

            self.prompt_layer_trend = nn.Linear(configs.d_model, configs.d_model)
            self.prompt_layer_season = nn.Linear(configs.d_model, configs.d_model)
            self.prompt_layer_noise = nn.Linear(configs.d_model, configs.d_model)

            for layer in (self.prompt_layer_trend, self.prompt_layer_season, self.prompt_layer_noise):
                layer.to(device=device)
                layer.train()
        else:
            self.out_layer_trend = nn.Linear(configs.d_model * self.patch_num+1, configs.pred_len)
            self.out_layer_season = nn.Linear(configs.d_model * self.patch_num+1, configs.pred_len)
            self.out_layer_noise = nn.Linear(configs.d_model * self.patch_num+1, configs.pred_len)


        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2_trend.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM, # causal language model
            r=16,
            lora_alpha=16,
            # target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",               # bias, set to only lora layers to train
            # modules_to_save=["classifier"],
        )
         
        self.gpt2_trend = get_peft_model(self.gpt2_trend, config)
        print_trainable_parameters(self.gpt2_trend)


        for layer in (self.gpt2_trend, self.in_layer_trend, self.out_layer_trend, \
                      self.in_layer_season, self.out_layer_season, self.in_layer_noise, self.out_layer_noise):
            layer.to(device=device)
            layer.train()

        for layer in (self.map_trend, self.map_season, self.map_resid):
            layer.to(device=device)
            layer.train()
        
        
        self.cnt = 0

        self.num_nodes = configs.num_nodes
        self.rev_in_trend = RevIn(num_features=self.num_nodes).to(device)
        self.rev_in_season = RevIn(num_features=self.num_nodes).to(device)
        self.rev_in_noise = RevIn(num_features=self.num_nodes).to(device)

        self.loss_func = configs.loss_func
        if self.loss_func == 'prob':
            # Output layers for Student's t-distribution parameters
            self.mu = nn.Linear(configs.pred_len, configs.pred_len)  # Mean
            self.sigma = nn.Linear(configs.pred_len, configs.pred_len)  # Scale (standard deviation)
            self.nu = nn.Linear(configs.pred_len, configs.pred_len)  # Degrees of freedom
        elif self.loss_func == 'negative_binomial':
            # Output layers for Negative Binomial parameters
            self.mu = nn.Linear(configs.pred_len, configs.pred_len)  # Mean
            self.alpha= nn.Linear(configs.pred_len, configs.pred_len) 
            
    @classmethod
    def load_pretrained_model(
        cls,
        device,
        cfg = None,
        repo_id="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir="./checkpoints/TEMPO_checkpoints"
    ):
        # Download the model checkpoint
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )

        # Download the config.json file
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            cache_dir=cache_dir
        )

        # Load the configuration file
        if cfg is None:
            cfg = OmegaConf.load(config_path)

        # Initialize the model
        model = cls(cfg, device)
        
        # Construct the full path to the checkpoint
        model_path = os.path.join(cfg.checkpoints, cfg.model_id)
        best_model_path = model_path + '_checkpoint.pth'
        print(f"Loading model from: {best_model_path}")
        
        # Load the state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        
        return model

        
    def store_tensors_in_dict(self, original_x, original_trend, original_season, original_noise, trend_prompts, season_prompts, noise_prompts):
        # Assuming prompts are lists of tuples       
        self.prompt_record_id += 1 
        for i in range(original_x.size(0)):
            self.prompt_record_plot[self.prompt_record_id + i] = {
                'original_x': original_x[i].tolist(),
                'original_trend': original_trend[i].tolist(),
                'original_season': original_season[i].tolist(),
                'original_noise': original_noise[i].tolist(),
                'trend_prompt': trend_prompts[i],
                'season_prompt': season_prompts[i],
                'noise_prompt': noise_prompts[i],
            }
        

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def select_prompt(self, summary, prompt_mask=None):
        # Build a matrix by stacking prompts keys from the dictionary
        prompt_key_matrix = torch.stack(tuple([self.prompt_key_dict[i] for i in self.prompt_key_dict.keys()]))
        
        # Normalize the prompt key matrix for similarity calculation (L2 normalization along feature dimension)
        prompt_norm = self.l2_normalize(prompt_key_matrix, dim=1) # Pool_size, C
        
        # Reshape the input summary tensor for processing
        summary_reshaped = summary.view(-1, self.patch_num)
        # Map the reshaped summary to the space defined by `summary_map`
        summary_mapped = self.summary_map(summary_reshaped)
        # Reshape the summary to match the expected dimensions for further operations
        summary = summary_mapped.view(-1, 768)
        # Normalize the reshaped summary embedding (L2 normalization along feature dimension)
        summary_embed_norm = self.l2_normalize(summary, dim=1)
        
        # Calculate the similarity between the normalized summary and prompt keys (via matrix multiplication)
        similarity = torch.matmul(summary_embed_norm, prompt_norm.t())
        
        # If a prompt mask is provided, use it, otherwise select top-k most similar prompts
        if not prompt_mask==None:
            idx = prompt_mask
        else:
            # Use `torch.topk` to find the top-k highest similarities along the given dimension (dim=1)
            topk_sim, idx = torch.topk(similarity, k=self.top_k, dim=1)
        
        # If no prompt mask was provided, calculate and update the occurrence of selected prompts in `prompt_record`
        if prompt_mask==None:
            # Flatten the selected index and count occurrences of each prompt index
            count_of_keys = torch.bincount(torch.flatten(idx), minlength=15)
            for i in range(len(count_of_keys)):
                self.prompt_record[f"id_{i}"] += count_of_keys[i].item()

        # Stack the values of selected prompts using their indices and remove any singleton dimensions
        prompt_value_matrix = torch.stack(tuple([self.prompt_value_dict[i] for i in self.prompt_value_dict.keys()]))
        batched_prompt_raw = prompt_value_matrix[idx].squeeze(1) # Shape: [Batch_size, Top_k, Length, C]
        
        # Reshape batched prompt into the desired dimensions: [batch_size, top_k * length, c]
        batch_size, top_k, length, c = batched_prompt_raw.shape # [16, 3, 5, 768]
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) 
        # Extract the normalized prompt keys corresponding to the selected indices
        batched_key_norm = prompt_norm[idx]
        # Ensure summary embedding has the appropriate shape for element-wise multiplication
        summary_embed_norm = summary_embed_norm.unsqueeze(1)
        # Compute element-wise similarity between summary and selected prompts
        sim = batched_key_norm * summary_embed_norm
        # Calculate the reduced similarity score by averaging across the batch
        reduce_sim = torch.sum(sim) / summary.shape[0]
        # Return the sorted tuple of selected prompts along with batched_prompt and reduce_sim
        selected_prompts = [tuple(sorted(row)) for row in idx.tolist()]
        # print("reduce_sim: ", reduce_sim)

        # Return the selected prompts, the computed reduced similarity, and the batched prompts
        return batched_prompt, reduce_sim, selected_prompts


    def get_norm(self, x, d = 'norm'):
        # if d == 'norm':
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        return x, means, stdev
    
    def get_patch(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # 4, 1, 420
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #4,1, 64, 16
        x = rearrange(x, 'b m n p -> (b m) n p') # 4, 64, 16 [batch_size, number of patches, patch_size]
        return x
    
    def get_emb(self, x, tokens=None, type = 'Trend'):
    # If no tokens are provided, process the input using the model's embeddings
        if tokens is None:
            if type == 'Trend':
            # Pass the input x through the GPT-2 model for trend type
                x = self.gpt2_trend(inputs_embeds =x).last_hidden_state
            # Pass the input x through the GPT-2 model for season type
            elif type == 'Season':
                x = self.gpt2_trend(inputs_embeds =x).last_hidden_state
            # Pass the input x through the GPT-2 model for residual type
            elif type == 'Residual':
                x = self.gpt2_trend(inputs_embeds =x).last_hidden_state
            return x  # Return the processed tensor
        
        # When tokens are provided, process them according to the type (trend/season/residual)
        else:
            #The variable x have the shape [batch_size, sequence_length, feature_dim], which is unpacked into a, b, c.
            [a,b,c] = x.shape 
          
            if type == 'Trend': 
                if self.pool:
                    # If pooling is enabled, select prompt and calculate similarity
                    prompt_x, reduce_sim, selected_prompts_trend = self.select_prompt(x, prompt_mask=None)
                    for selected_prompt_trend in selected_prompts_trend:
                        # Record the occurrence of selected prompt
                        self.prompt_record_trend[selected_prompt_trend] = self.prompt_record_trend.get(selected_prompt_trend, 0) + 1
                    selected_prompts = selected_prompts_trend
                else:
                    # If no pooling, get the word token embeddings for the provided tokens
                    prompt_x = self.gpt2_trend.wte(tokens) #wte ("word token embeddings") is func of GPT-2 (part of the pre-trained transformer model)
                    #wte represents a layer that maps discrete token indices into vector embeddings.      
                    
                    prompt_x = prompt_x.repeat(a,1,1)
                    # Pass the embeddings through a transformation layer for 'Trend'
                    prompt_x = self.prompt_layer_trend(prompt_x)
                # Concatenate patch and prompt embeddings - Concatenate the prompt embeddings with the original x tensor along the sequence dimension    
                x = torch.cat((prompt_x, x), dim=1)
                
            elif type == 'Season':
                if self.pool:
                    prompt_x, reduce_sim, selected_prompts_season = self.select_prompt(x, prompt_mask=None)
                    for selected_prompt_season in selected_prompts_season:
                        self.prompt_record_season[selected_prompt_season] = self.prompt_record_season.get(selected_prompt_season, 0) + 1
                    selected_prompts = selected_prompts_season
                else:
                    prompt_x = self.gpt2_trend.wte(tokens)
                    prompt_x = prompt_x.repeat(a,1,1)
                    prompt_x = self.prompt_layer_season(prompt_x)
                # Concatenate the prompt embeddings with the original x tensor along the sequence dimension
                x = torch.cat((prompt_x, x), dim=1)
                # x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state
                
            elif type == 'Residual':
                if self.pool:
                    prompt_x, reduce_sim, selected_prompts_resid = self.select_prompt(x, prompt_mask=None)
                    for selected_prompt_resid in selected_prompts_resid:
                        self.prompt_record_residual[selected_prompt_resid] = self.prompt_record_residual.get(selected_prompt_resid, 0) + 1
                    selected_prompts = selected_prompts_resid
                else:
                    prompt_x = self.gpt2_trend.wte(tokens)
                    prompt_x = prompt_x.repeat(a,1,1)
                    prompt_x = self.prompt_layer_noise(prompt_x)
                # prompt_x, reduce_sim_trend = self.select_prompt(x, prompt_mask=None)
                # Concatenate the prompt embeddings with the original x tensor along the sequence dimension
                x = torch.cat((prompt_x, x), dim=1)

            # Return the processed tensor, with potential reduction in similarity and selected prompts for pooling    
            if self.pool:
                return x, reduce_sim, selected_prompts
            else:
                return x
            
    def create_image(self, x_local, type, dataset, show_plot=False):
        """
        Plots a time series and returns it as a normalized image tensor.

        Args:
            x_local: Time series data - can be either a NumPy array or PyTorch tensor
            type (str): One of ['Trend', 'Season', 'Residual']
            dataset (str): Dataset name to include in filename
            show_plot (bool): If True, will display the plot using plt.show()
            
        Returns:
            Tensor: Image tensor (e.g., for CLIP)
        """

        save_dir = './plot_pics'
        os.makedirs(save_dir, exist_ok=True)

        # Initialize counters if they don't exist
        if not hasattr(self, 'trend_plot_counter'):
            self.trend_plot_counter = 0
        if not hasattr(self, 'season_plot_counter'):
            self.season_plot_counter = 0
        if not hasattr(self, 'residual_plot_counter'):
            self.residual_plot_counter = 0

        images = []

        # Check if x_local is a NumPy array or a single value
        if isinstance(x_local, np.ndarray) or np.isscalar(x_local):
            # Create a batch of size 1 to maintain compatibility
            if np.isscalar(x_local):
                # If it's a single value, convert to a 1D array
                time_series = np.array([x_local])
            else:
                # If it's already an array, use it directly
                time_series = x_local
                
            time_steps = range(len(time_series))
            
            fig, ax = plt.subplots(figsize=(5, 5))
            
            # Plot and update based on type
            if type == 'Trend':
                counter = self.trend_plot_counter
                label = f"{dataset} Plot {type}_{counter + 1}"
                filename = f"{label}.png"
                self.trend_plot_counter += 1
            elif type == 'Season':
                counter = self.season_plot_counter
                label = f"{dataset} Plot {type}_{counter + 1}"
                filename = f"{label}.png"
                self.season_plot_counter += 1
            else:  # Residual or any fallback
                counter = self.residual_plot_counter
                label = f"{dataset} Plot {type}_{counter + 1}"
                filename = f"{label}.png"
                self.residual_plot_counter += 1
            
            ax.bar(time_steps, time_series, label=label)
            ax.set_xlabel("Features")
            ax.set_ylabel("Features Values")
            # ax.set_title(label)
            # ax.legend()
            
            show_plot = False 
            if show_plot:
                plt.show()
            
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path)
            
            # Save into buffer
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            buf.close()
            plt.close(fig)
            
            # Preprocess the image using the vision encoder's preprocess function
            image_tensor = self.vision_encoder_preprocess(image).to(self.device)
            images.append(image_tensor)
        else:
            # Original code for PyTorch tensors
            for i in range(x_local.shape[0]):  # Iterate over the batch
                time_series = x_local[i].squeeze().cpu().detach().numpy()
                time_steps = range(len(time_series))

                fig, ax = plt.subplots(figsize=(5, 5))

                # Plot and update based on type
                if type == 'Trend':
                    counter = self.trend_plot_counter
                    label = f"{dataset} Plot {type}_{counter + 1}"
                    filename = f"{label}.png"
                    self.trend_plot_counter += 1
                elif type == 'Season':
                    counter = self.season_plot_counter
                    label = f"{dataset} Plot {type}_{counter + 1}"
                    filename = f"{label}.png"
                    self.season_plot_counter += 1
                else:  # Residual or any fallback
                    counter = self.residual_plot_counter
                    label = f"{dataset} Plot {type}_{counter + 1}"
                    filename = f"{label}.png"
                    self.residual_plot_counter += 1

                ax.bar(time_steps, time_series, label=label)
                ax.set_xlabel("Features")
                ax.set_ylabel("Features Values")
                # ax.set_title(label)
                # ax.legend()
                
                if show_plot:
                    plt.show()

                save_path = os.path.join(save_dir, filename)
                fig.savefig(save_path)

                # Save into buffer
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf).convert("RGB")
                buf.close()
                plt.close(fig)
                
                # Preprocess the image using the vision encoder's preprocess function
                image_tensor = self.vision_encoder_preprocess(image).to(self.device)
                images.append(image_tensor)

        ans = torch.stack(images)
        return ans
   
    def vision_embed(self, image, type='Trend'):
        # Process image and compute its embedding
        with torch.no_grad():
            if self.vis_encoder_name == "CLIP":
                image_embed_vec = self.vision_encoder.encode_image(image)
            elif self.vis_encoder_name == "SigLIP":
                outputs = self.vision_encoder(image)
                image_embed_vec = outputs.last_hidden_state.mean(dim=1)
            else:
                image_embed_vec = self.vision_encoder(image)
        image_embed_vec = image_embed_vec.to(self.vis_layer_trend.weight.dtype)
        return image_embed_vec
    

    @torch.no_grad()
    def compute_vision_embeddings(self, x, save_dir="/Pics_embed", data = None):
        """Computes only the vision embeddings without running the full forward process."""
        os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
        
        # B, L, M = x.shape  # Batch, Length, Features [[B,L,M]]
        if self.use_components == 0:
            trend_image = self.create_image(x, 'Trend' , data)
            # season_image = self.create_image(x, 'Season', data)
            # noise_image = self.create_image(x, 'Residual', data)
            
        else:
            x = self.rev_in_trend(x, 'norm')
            x = self.moving_avg(x)
            # Compute trend, season, and noise components
            x_cpu = x.cpu().numpy()
            x_cpu = x_cpu.flatten()
            components = STL(x_cpu, period=24).fit()
            trend_local = components.trend
            season_local = components.seasonal
            noise_local = components.resid

            # trend_local = self.moving_avg(x)
            # trend_local = self.map_trend(trend_local.squeeze(2)).unsqueeze(2)
            # season_local = x - trend_local
            # season_local = self.map_season(season_local.squeeze(2)).unsqueeze(2)
            # noise_local = x - trend_local - season_local
            
            # -------------if vis will be local for every patch (and not global)--------------
            # trend = self.get_patch(trend_local) # 4, 64, 16
            # season = self.get_patch(season_local)
            # noise = self.get_patch(noise_local)

            # Create images from the components
            trend_image = self.create_image(trend_local, 'Trend' , data)
            season_image = self.create_image(season_local, 'Season', data)
            noise_image = self.create_image(noise_local, 'Residual', data)
        
        # Compute vision embeddings
        trend_embed = self.vision_embed(trend_image, 'Trend')
        # season_embed = self.vision_embed(season_image, 'Season')
        # noise_embed = self.vision_embed(noise_image, 'Residual')
            
        return trend_embed
        # return trend_embed, season_embed, noise_embed




    def forward(self, x, itr=0, trend=None, season=None, noise=None, test=False):
        B, L, M = x.shape # 4, 512, 1
        x = self.rev_in_trend(x, 'norm')
        original_x = x
        # Moving average for trend
        trend_local = self.moving_avg(x)
        # Map trend
        trend_local = self.map_trend(trend_local.squeeze(2)).unsqueeze(2)
        # Calculate season
        season_local = x - trend_local
        # Map season
        season_local = self.map_season(season_local.squeeze(2)).unsqueeze(2)
        # Calculate noise
        noise_local = x - trend_local - season_local
        
        if trend is not None:
            trend, means_trend, stdev_trend = self.get_norm(trend)
            season, means_season, stdev_season = self.get_norm(season)
            noise, means_noise, stdev_noise = self.get_norm(noise)
            trend_local_l = criterion(trend, trend_local)
            season_local_l = criterion(season, season_local)
            noise_local_l = criterion(noise, noise_local)
            loss_local = trend_local_l + season_local_l + noise_local_l 
            #import ipdb; ipdb.set_trace()
            if test:
                print("trend local loss:", torch.mean(trend_local_l))
                print("Season local loss", torch.mean(season_local_l))
                print("noise local loss", torch.mean(noise_local_l))

        trend = self.get_patch(trend_local) # 4, 64, 16
        season = self.get_patch(season_local)
        noise = self.get_patch(noise_local)
        
        if self.vision:
            # creating plot image of each component
            trend_image = self.create_image(trend_local)
            season_image = self.create_image(season_local)
            noise_image = self.create_image(noise_local)

        # in_layer_trend: patch_size ---> d_model
        trend = self.in_layer_trend(trend) # 4, 64, 768  [batch size, number of patches, patch_size ---> d_model]
        if self.is_gpt and self.prompt == 1:
            if self.pool:
                trend, reduce_sim_trend, trend_selected_prompts = self.get_emb(trend, self.gpt2_trend_token['input_ids'], 'Trend')
            else:
                # trend is a concatination of the prompt embed (prompt_x) and the patch embed (x)
                trend = self.get_emb(trend, self.gpt2_trend_token['input_ids'], 'Trend')
                if self.vision:
                    trend = self.vision_embed(trend, trend_image, 'Trend')
                    self.save_embedding(trend, 'Trend', folder_id)
        else:
            trend = self.get_emb(trend)
            if self.vision:
                trend = self.vision_embed(trend, trend_image, 'Trend')
                self.save_embedding(trend, 'Trend', folder_id)

        season = self.in_layer_season(season) # 4, 64, 768
        if self.is_gpt and self.prompt == 1:
            if self.pool:
                season, reduce_sim_season, season_selected_prompts = self.get_emb(season, self.gpt2_season_token['input_ids'], 'Season')
            else:
                season = self.get_emb(season, self.gpt2_season_token['input_ids'], 'Season')
                if self.vision:
                    season = self.vision_embed(season, season_image, 'Season')
                    self.save_embedding(season, 'Season', folder_id)
        else:
            season = self.get_emb(season)
            if self.vision:
                season = self.vision_embed(season, season_image, 'Season')
                self.save_embedding(season, 'Season', folder_id)

        noise = self.in_layer_noise(noise)
        if self.is_gpt and self.prompt == 1:
            if self.pool:
                noise, reduce_sim_noise, noise_selected_prompts = self.get_emb(noise, self.gpt2_residual_token['input_ids'], 'Residual')
            else:
                noise = self.get_emb(noise, self.gpt2_residual_token['input_ids'], 'Residual')
                if self.vision:
                    noise = self.vision_embed(noise, noise_image, 'Residual')
                    self.save_embedding(noise, 'Residual', folder_id)

        else:
            noise = self.get_emb(noise)
            if self.vision:
                noise = self.vision_embed(noise, noise_image, 'Residual')
                self.save_embedding(noise, 'Residual', folder_id)

        # print(noise_selected_prompts)

        # self.store_tensors_in_dict(original_x, trend_local, season_local, noise_local, trend_selected_prompts, season_selected_prompts, noise_selected_prompts)
        
        x_all = torch.cat((trend, season, noise), dim=1)
        
        # print('vis_saved')
        # return 

        x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state 
        vision_token_len = 1
        
        if self.prompt == 1:
            if self.vision:
                trend  = x[:, :self.token_len+self.patch_num+vision_token_len, :]  
                season  = x[:, self.token_len+self.patch_num+vision_token_len:2*self.token_len+2*self.patch_num+2*vision_token_len, :]  
                noise = x[:, 2*self.token_len+2*self.patch_num+2*vision_token_len:, :]
                if self.use_token == 0:
                    trend = trend[:, self.token_len+vision_token_len:, :]
                    season = season[:, self.token_len+vision_token_len:, :]
                    noise = noise[:, self.token_len+vision_token_len:, :]
            else:
                trend  = x[:, :self.token_len+self.patch_num, :]  
                season  = x[:, self.token_len+self.patch_num:2*self.token_len+2*self.patch_num, :]  
                noise = x[:, 2*self.token_len+2*self.patch_num:, :]
                if self.use_token == 0:
                    trend = trend[:, self.token_len:, :]
                    season = season[:, self.token_len:, :]
                    noise = noise[:, self.token_len:, :]    
        else:
            trend  = x[:, :self.patch_num, :]  
            season  = x[:, self.patch_num:2*self.patch_num, :]  
            noise = x[:, 2*self.patch_num:, :] 
            
        trend = self.out_layer_trend(trend.reshape(B*M, -1)) # 4, 96
        trend = rearrange(trend, '(b m) l -> b l m', b=B) # 4, 96, 1
        season = self.out_layer_season(season.reshape(B*M, -1)) # 4, 96
        # print(season.shape)
        season = rearrange(season, '(b m) l -> b l m', b=B) # 4, 96, 1
        # season = season * stdev_season + means_season
        noise = self.out_layer_noise(noise.reshape(B*M, -1)) # 4, 96
        noise = rearrange(noise, '(b m) l -> b l m', b=B)
        # noise = noise * stdev_noise + means_noise
        
        outputs = trend + season + noise #season #trend # #+ noise
        # outputs = outputs * stdev + means
        outputs = self.rev_in_trend(outputs, 'denorm')
        # if self.pool:
        #     return outputs, loss_local #loss_local - reduce_sim_trend - reduce_sim_season - reduce_sim_noise
        if self.loss_func == 'prob':
            outputs = rearrange(outputs, 'b l m-> b m l', b=B).squeeze()

            mu = self.mu(outputs)
            sigma = F.softplus(self.sigma(outputs)) + 1e-6  # Ensure scale is positive
            nu = F.softplus(self.nu(outputs)) + 2   # Ensure degrees of freedom > 2
            if test:
                return (mu, sigma, nu), None
            return (mu, sigma, nu), loss_local
        elif self.loss_func == 'negative_binomial':
            mu = F.softplus(self.mu(x)) + 1e-4  # Ensure mean is positive
            alpha = F.softplus(self.alpha(x)) + 1e-4  # Ensure dispersion is positive
            if test:
                return (mu.permute(0,2,1), alpha.permute(0,2,1)), None  # Return to [Batch, Output length, Channel]
            else:
                return (mu.permute(0,2,1), alpha.permute(0,2,1)), loss_local
        
        if test:
            return outputs, None
        
        return outputs, loss_local
    
    

    def predict(self, x, pred_length=96):
        """
        Predict using the TEMPO model.
        
        Args:
        - x: Input time series data (shape: [B, L, M])
        
        Returns:
        - Predicted output
        """
        self.eval()  # Set the model to evaluation mode

        x = torch.FloatTensor(x).unsqueeze(0).unsqueeze(2).to(self.device)  # Shape: [1, 336, 1]
        x = self.rev_in_trend(x, 'norm')
        
        B, L, M = x.shape
        target_length = self.seq_len  # Maximum supported length
        
        if L > target_length:
            warnings.warn(f"Input length {L} is larger than the maximum supported length of {target_length}. "
                          f"This may influence performance. Cutting the input to the last {target_length} time steps.")
            x = x[:, -target_length:, :]
        elif L < target_length:
            pad_length = target_length - L
            if pad_length <= L:
                # Pad by repeating the time series
                x_padded = torch.cat([x] * (target_length // L + 1), dim=1)[:, :target_length, :]
            else:
                # Pad with zeros at the beginning
                padding = torch.zeros(B, pad_length, M, device=x.device)
                x_padded = torch.cat([padding, x], dim=1)
            
            x = x_padded
            warnings.warn(f"Input length {L} is smaller than the required length of {target_length}. "
                          f"The time series has been {'repeated' if pad_length <= L else 'zero-padded'} to reach the required length.")
        
        # Ensure x is on the same device as the model
        x = x.to(self.device)
       
        
        
        
        # with torch.no_grad():
        #     outputs, _ = self.forward(x, test=True)        
        # # Extract the predicted values
        # predicted_values = outputs.squeeze().numpy()[-pred_length:]
        # return predicted_values

        with torch.no_grad():
            current_input = x.clone()
            all_predictions = []
            
            while len(all_predictions) < pred_length:
                # Forward pass
                outputs, _ = self.forward(current_input, test=True)
                outputs = self.rev_in_trend(outputs, 'denorm')
                step_size = outputs.shape[1]
                # Extract the predicted values
                predicted_values = outputs.cpu().squeeze().numpy()[-step_size:]
                
                # Append to all predictions
                all_predictions.extend(predicted_values)
                
                # Update the input for the next iteration
                new_sequence = np.concatenate([current_input.cpu().squeeze().numpy()[step_size:], predicted_values])
                current_input = torch.FloatTensor(new_sequence).unsqueeze(0).unsqueeze(2)
        # Trim to the desired length
        return np.array(all_predictions[:pred_length])