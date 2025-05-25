ts_by_feature=1

if [ "$ts_by_feature" -eq 1 ]; then
  enc_in=405
  seq_len=61
else
  enc_in=61
  seq_len=405
fi
  
python -u train_TEMPO.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets Heartbeat\
  --num_classes 2 \
  --seq_len $seq_len \
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
  --ts_by_feature $ts_by_feature \
  --train_epochs 5 \
  --patience 5 \
  --enc_in $enc_in \
  --vision 1 \
  --create_offline_vision 0 \
  --use_components 0 \
  --target_data Heartbeat \
  --prompt 1 \
  --take_vis_by_feature 1 \
  --all_components 0 \
  --patch_size 1 \
  --stride 4


  