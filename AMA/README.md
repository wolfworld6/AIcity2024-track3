# Preprocess

## Prepare
### 1.request
pip install -r requirements.txt
### 2.NMS
```
cd ./libs/utils
python setup.py install --user
cd ../..
```

## Data Preparation

1. AIcity Video Features Preparation
Follow the step [preprocess](preprocess/README.md)

2. Change the aicity_ego_vitl.yaml

&emsp;In aicity_ego_vitl.yaml, you can specify annotation json_file, video features, training split and validation split, e.t.c.


## Train & val

### 1. specify annotation json_file with "label_A1-train_A1-val.json", "feat_folder", then run:
```
python train.py ./configs/aicity_ego_vitl.yaml --output ckpt/mae2_f16_e20_1024_ide_4h_w9_feats_ego4d_vitl_f16_8h_9k_track3_crop_A1_train_A2_val/
```

### 2. generate act_pre_ego_crop_AMA_mae2_f16_f16_8h_9k_submmit.csv, specify annotation json_file with "label_submit_B.json", "pre_nms_topk: 3000", "max_seg_num: 150" then run:
```
python eval.py ./configs/aicity_ego_vitl.yaml ckpt/aicity_ego_vitl_ckpt/mae2_f16_e20_1024_ide_4h_w9_feats_ego4d_vitl_f16_8h_9k_track3_crop_A1_train_A2_val/

```


## Submission
Follow the step [post_process](../post_process/README.md)