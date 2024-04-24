# Feature Extraction
Most of the code comes from [VideoMAEv2: Scaling Video Masked Autoencoders with Dual Masking.](https://github.com/OpenGVLab/VideoMAEv2/tree/master)

## Prepare
```
export BASE_DIR=/xxx/AIcity2024-track3
cd $BASE_DIR/feature_extraction
```

## Installation
Please follow the instrctions in [INSTALL.md](../feature_extraction/docs/INSTALL.md)

## Pretrain Weights
We choose an open source pretrained weight in final competition. The weight needs to apply for download, so we provide link as follows to get the weight directly. 
<!-- You can download the weight and put it into *__weights__* folder of feature extraction. -->
    
* Download [vit_g_hybrid_pt_1200e_k710_ft](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/videomaev2/vit_g_hybrid_pt_1200e_k710_ft.pth) and put it into `$BASE_DIR/feature_extraction/weights`.

## Fine-tune model on official data
To fine-tune model (VideoMAEv2 with pretrain weight mentioned above) on the official dataset A1 with *__8xA100-40G__*, you can use the following command:

```
bash scripts/finetune/track3_vit_g_A1_ft.sh
``` 

Also we provide finetuned models: <a href="https://huggingface.co/wolfutopia/videomae-v2_finetune_aicity"> videomae-v2_finetune_aicity.pth 🤗</a>&nbsp;. You can download it and put it into `$BASE_DIR/feature_extraction/weights` for feature extraction.

## Inference to extract video features
You can extract video features by running:
```
# extract video feats for A1
python inference_video_feature_vitg.py \
    --video_dir $BASE_DIR/data/crop_videos/A1 \
    --ckpt_pth weights/videomae-v2_finetune_aicity.pth \
    --output_dir $BASE_DIR/data/extracted_features/A1

# extract video feats for A2
python inference_video_feature_vitg.py \
    --video_dir $BASE_DIR/data/crop_videos/A2 \
    --ckpt_pth weights/videomae-v2_finetune_aicity.pth \
    --output_dir $BASE_DIR/data/extracted_features/A2
```

Also we provide extracted feats for dataset A1&A2 here: [extracted feats](https://drive.google.com/drive/folders/1y19_yF-mDi_SoHmPVYF0mfKepLcXufwB?usp=sharing) and put the into `$BASE_DIR/data/extracted_features/`.
