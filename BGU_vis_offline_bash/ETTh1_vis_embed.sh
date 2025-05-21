#!/bin/bash
#SBATCH --job-name=TEMPO_Embed        # Job name
#SBATCH --output=output_TEMPO_embed_%A_%a.txt   # Output log
#SBATCH --nodes=1                     # Single node  
#SBATCH --ntasks=1                    # Single CPU task
#SBATCH --mem=20G                      # RAM allocation
#SBATCH --cpus-per-task=64             # Number of CPU cores
#SBATCH --gres=gpu:1                   # GPU allocation
#SBATCH -p gpu                         # GPU partition
#SBATCH --time=12:00:00                # Max runtime
#SBATCH --qos=gpu-8                    # Allow up to 8 GPUs

export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0

# Set hyperparameters
model="TEMPO"  # Only TEMPO is used
batch_size=1
train_epochs=10
learning_rate=0.001
patience=5
save_dir="/home/arielsi/VisionaryTimes/Pics_embed"
seq_len=512 


# Define dataset arguments
datasets="ETTh1"
target_data="ETTh1"
eval_data="ETTh1"

# Create log directories
mkdir -p logs/$model
log_file="logs/$model/generate_vis_embed_${datasets}.log"
echo "Logging to: $log_file"

# Run the Python script
python generate_vis_embed_offline.py \
    --config_path "./configs/multiple_datasets.yml" \
    --batch_size $batch_size \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --model $model \
    --save_dir $save_dir \
    --datasets $datasets \
    --target_data $target_data \
    --eval_data $eval_data \
    --model "TEMPO" \
    --batch_size 1 \
    --train_epochs 10 \
    --learning_rate 0.001 \
    --patience 5 \
    --seq_len 512 \
    --datasets ETTh1\
    --target_data ETTh1\
    --eval_data ETTh1\
    --use_components 1\
    --show_plot 0\
    --vis_encoder_name "DeiT-Tiny"\
    --vis_encoder_dim 192 \
    --vision 1 \

