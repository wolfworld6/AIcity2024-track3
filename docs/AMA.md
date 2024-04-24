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

## Data Prepare

Follow the step [preprocess](../docs/PREPROCESS.md) and [feature extraction](../docs/FEATURE_EXTRACTION.md).


# Train

```
python train.py ./configs/aicity_ego_vitl.yaml --output ckpt/mae2_f16_e20_1024_ide_4h_w9_feats_ego4d_vitl_f16_8h_9k_track3_crop_A1_train_A2_val/
```

# val
* Download the ckpt [finetuned AMA](https://drive.google.com/drive/folders/13lT2GcsI-VK5z--8rirgR7nF6sZeiXo7?usp=sharing) and put it into `$BASE_DIR/AMA/ckpt/aicity_ego_vitl_ckpt/`

 * Generate csv for post process.

<!-- specify annotation json_file with `"label_submit_B.json"`, `"pre_nms_topk: 3000"`, `"max_seg_num: 150"` then run: -->

```
python eval.py ./configs/aicity_ego_vitl_deploy.yaml ckpt/aicity_ego_vitl_ckpt/mae2_f16_e20_1024_ide_4h_w9_feats_ego4d_vitl_f16_8h_9k_track3_crop_A1_train_A2_val/ --output_csv $BASE_DIR/post_process/act_pre_ego_crop_AMA_mae2_f16_f16_8h_9k_submmit.csv
```

