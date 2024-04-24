# Preparation
## Install environment
```
export BASE_DIR=/xxx/AIcity2024-track3
cd $BASE_DIR/AMA

pip install -r requirements.txt
```

```
cd ./libs/utils
python setup.py install --user
cd ../..
```

## data

1. AIcity Video Features Preparation

    Follow the step [preprocess](../docs/PREPROCESS.md).

2. Change the aicity_ego_vitl.yaml

    In aicity_ego_vitl.yaml, you can specify annotation json_file, video features, training split and validation split, e.t.c.


# Train
 specify annotation json_file with "label_A1-train_A1-val.json", "feat_folder", then run:
```
python train.py ./configs/aicity_ego_vitl.yaml --output ckpt/mae2_f16_e20_1024_ide_4h_w9_feats_ego4d_vitl_f16_8h_9k_track3_crop_A1_train_A2_val/
```

# val
* download the ckpt [finetuned AMA](https://drive.google.com/drive/folders/13lT2GcsI-VK5z--8rirgR7nF6sZeiXo7?usp=sharing), put it into `$BASE_DIR/AMA/ckpt/aicity_ego_vitl_ckpt/`

 * generate `act_pre_ego_crop_AMA_mae2_f16_f16_8h_9k_submmit.csv`

<!-- specify annotation json_file with `"label_submit_B.json"`, `"pre_nms_topk: 3000"`, `"max_seg_num: 150"` then run: -->

```
python eval.py ./configs/aicity_ego_vitl_deploy.yaml ckpt/aicity_ego_vitl_ckpt/mae2_f16_e20_1024_ide_4h_w9_feats_ego4d_vitl_f16_8h_9k_track3_crop_A1_train_A2_val/ --output_csv ../post_process/act_pre_ego_crop_AMA_mae2_f16_f16_8h_9k_submmit.csv
```


## Submission
Follow the step [post_process](../docs/POST_PROCESS.md).
