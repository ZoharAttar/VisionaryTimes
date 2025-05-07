  ts_by_feature=1

if [ "$ts_by_feature" -eq 1 ]; then
  enc_in=152
  seq_len=3
else
  enc_in=3
  seq_len=152
fi
  
    python -u train_TEMPO.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets Handwriting\
  --num_classes 26 \
  --seq_len $seq_len \
  --pred_len 0 \
  --model_id Handwriting'_'TEMPO'_' \
  --model TEMPO \
  --data Handwriting \
  --e_layers 3 \
  --batch_size 8 \
  --d_model 768 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.001 \
  --ts_by_feature $ts_by_feature \
  --train_epochs 10 \
  --patience 5 \
  --enc_in $enc_in \
  --patch_size 8