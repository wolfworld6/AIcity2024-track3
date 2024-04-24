#!/usr/bin/env bash
MASTER_PORT=$((12000 + $RANDOM % 20000))
OUTPUT_DIR='./weights/videomae-v2_finetune_aicity.pth'
DATA_PATH='../data/splited_videos'
MODEL_PATH='./weights/vit_g_hybrid_pt_1200e_k710_ft.pth'


# batch_size can be adjusted according to the graphics card
python -m torch.distributed.launch \
        --nproc_per_node=8 --master_port ${MASTER_PORT} \
        --nnodes=1  --node_rank=0 --master_addr=localhost run_class_finetuning.py \
        --model vit_giant_patch14_224 \
        --data_set Track3 \
        --nb_classes 16 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 3 \
        --epochs 35 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \

