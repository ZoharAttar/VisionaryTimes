  python -u train_TEMPO.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets Heartbeat\
  --num_classes 2 \
  --seq_len 405 \
  --pred_len 0 \
  # --is_training 1 \
  # --root_path ./dataset/Heartbeat/ \
  # --model_id Heartbeat \
  --model_id Heartbeat'_'TEMPO'_' \
  --model TEMPO \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  # --top_k 3 \
  # --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10
  