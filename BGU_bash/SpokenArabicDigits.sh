python -u train_TEMPO.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets SpokenArabicDigits\
  --num_classes 2 \
  --seq_len 405 \
  --pred_len 0 \
  # --is_training 1 \
  # --root_path ./dataset/SpokenArabicDigits/ \
  # --model_id SpokenArabicDigits \
  --model_id SpokenArabicDigits'_'TEMPO'_' \
  --model TEMPO \
  --data SpokenArabicDigits \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  # --top_k 3 \
  # --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --ts_by_feature 0 \
  --train_epochs 100 \
  --patience 10

