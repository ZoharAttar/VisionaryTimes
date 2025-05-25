ts_by_feature=1

if [ "$ts_by_feature" -eq 1 ]; then
  enc_in=93
  seq_len=13
else
  enc_in=13
  seq_len=93
fi
  
python -u generate_vis_embed_offline.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets SpokenArabicDigits\
  --num_classes 10 \
  --seq_len $seq_len \
  --pred_len 0 \
  --model_id SpokenArabicDigits'_'TEMPO'_' \
  --model TEMPO \
  --data SpokenArabicDigits \
  --e_layers 3 \
  --batch_size 1 \
  --d_model 768 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.001 \
  --ts_by_feature $ts_by_feature \
  --train_epochs 10 \
  --patience 5 \
  --enc_in $enc_in \
  --create_offline_vision 1 \
  --vision 1 \
  --use_components 0 \
  --target_data SpokenArabicDigits \
  --vis_encoder_dim 192 \
  --vis_encoder_name DeiT-Tiny
  