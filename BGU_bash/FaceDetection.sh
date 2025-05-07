ts_by_feature=1

if [ "$ts_by_feature" -eq 1 ]; then
  enc_in=62
  seq_len=144
else
  enc_in=144
  seq_len=62
fi
  
    python -u train_TEMPO.py \
  --task_name classification \
  --config_path ./configs/multiple_datasets.yml \
  --datasets FaceDetection\
  --num_classes 2 \
  --seq_len $seq_len \
  --pred_len 0 \
  --model_id FaceDetection'_'TEMPO'_' \
  --model TEMPO \
  --data FaceDetection \
  --e_layers 3 \
  --batch_size 8 \
  --d_model 768 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.001 \
  --ts_by_feature $ts_by_feature \
  --train_epochs 10 \
  --patience 5 \
  --enc_in $enc_in