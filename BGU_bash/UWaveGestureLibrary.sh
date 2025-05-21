ts_by_feature=1

if [ "$ts_by_feature" -eq 1 ]; then
  enc_in=315
  seq_len=3
else
  enc_in=3
  seq_len=315
fi
  
    python -u train_TEMPO.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets UWaveGestureLibrary\
  --num_classes 8 \
  --seq_len $seq_len \
  --pred_len 0 \
  --model_id UWaveGestureLibrary'_'TEMPO'_' \
  --model TEMPO \
  --data UWaveGestureLibrary \
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
  --vision 0 \
  --create_offline_vision 0 \
  --use_components 0 \
  --target_data UWaveGestureLibrary \
  --prompt 1 \
  --take_vis_by_feature 1 \
  --all_components 0 \
  --patch_size 3 \
  --stride 1
