# Example for Test on Dataset B
```
export BASE_DIR=/xxx/AIcity2024-track3
```
## 1. Dataset Preparation

We recommend placing and treating dataset *__B__* like dataset *__A2__*

* crop the human body of the input videos

    ```
    cd $BASE_DIR/preprocess

    python yolov5/driver_tracking.py --vid_path $BASE_DIR/data/raw_videos/B  --out_file $BASE_DIR/data/crop_videos/B
    ```

* generate json files
    ```
    python get_json_for_data_A2.py --video_path $BASE_DIR/data/crop_videos/B --output_path $BASE_DIR/data/label_submit_B.json
    ```

## 2. Feautre Extraction
* Download weights <a href="https://huggingface.co/wolfutopia/videomae-v2_finetune_aicity"> 🤗</a>&nbsp;.
* Extracting video features of B dataset using trained weights.
```
cd $BASE_DIR/feature_extraction

python inference_video_feature_vitg.py \
    --video_dir $BASE_DIR/data/crop_videos/B \
    --ckpt_pth trained-weights \
    --output_dir $BASE_DIR/data/extracted_features/B
```

## 3. AMA
<a id="TAD"></a>

1. Download weights [finetuned AMA](https://drive.google.com/drive/folders/13lT2GcsI-VK5z--8rirgR7nF6sZeiXo7?usp=sharing), put it into `$BASE_DIR/AMA/ckpt/aicity_ego_vitl_ckpt/`

2. `cd $BASE_DIR/AMA` and modeify `configs/aicity_ego_vitl_deploy.yaml` :


 ```
 json_file: $BASE_DIR/data/label_submit_B.json,
 feat_folder: $BASE_DIR/data/extracted_features/B,
 ```
 3. run

```
python eval.py ./configs/aicity_ego_vitl_deploy.yaml ckpt/aicity_ego_vitl_ckpt/mae2_f16_e20_1024_ide_4h_w9_feats_ego4d_vitl_f16_8h_9k_track3_crop_A1_train_A2_val/ --output_csv $BASE_DIR/post_process/submmit_B.csv
```

## 4. Post Process

The `submmit_B.csv` from  [3. AMA](#TAD)
```
cd $BASE_DIR/post_process

python generate_txt.py --csv_path submmit_B.csv --out_file submit/submmit_B.txt
```
