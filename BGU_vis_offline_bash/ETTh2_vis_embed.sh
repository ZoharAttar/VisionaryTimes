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
# save_dir="/home/arielsi/VisionaryTimes/Pics_embed"
save_dir="/home/arielsi/VisionaryTimes/plot_pics_no_components"
pred_len=96
seq_len=512 


# Define dataset arguments
datasets="ETTh2"
target_data="ETTh2"
eval_data="ETTh2"

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
    --vision 1 \
    --save_dir $save_dir \
    --datasets $datasets \
    --target_data $target_data \
    --eval_data $eval_data \
    --create_offline_vision 1\
    --model "TEMPO" \
    --batch_size 1 \
    --train_epochs 10 \
    --learning_rate 0.001 \
    --patience 5 \
    --pred_len 96 \
    --seq_len 512 \
    --datasets ETTh2\
    --target_data ETTh2\
    --eval_data ETTh2\
    --use_components 0\
    --show_plot 1\
    # >> $log_file 2>&1