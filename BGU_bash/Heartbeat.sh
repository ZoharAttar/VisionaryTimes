  python -u train_TEMPO.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets Heartbeat\
  --num_classes 2 \
  --seq_len 405 \
  --pred_len 0 \
  --model_id Heartbeat'_'TEMPO'_' \
  --model TEMPO \
  --data Heartbeat \
  --e_layers 3 \
  --batch_size 4 \
  --d_model 768 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.001 \
  --ts_by_feature 1 \
  --train_epochs 10 \
  --patience 5
  