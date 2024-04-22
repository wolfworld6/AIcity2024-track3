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
We choose an open source pretrained weight in final competition. The weight needs to apply for download, so we provide network disk as follows to get the weight directly. You can download the weight and put it into *__weights__* folder of feature extraction.
    
* vit_g_hybrid_pt_1200e_k710_ft, from https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md, disk download: [Link](https://drive.google.com/drive/folders/1y19_yF-mDi_SoHmPVYF0mfKepLcXufwB?usp=sharing)

## Fine-tune model on official data
To fine-tune model (VideoMAEv2 with pretrain weight mentioned above) on the official dataset A1 with *__8xA100-40G__*, you can use the following command:

```
bash scripts/finetune/track3_vit_g_A1_ft.sh
``` 

Also we provide finetuned models here: [Link](https://drive.google.com/drive/folders/1y19_yF-mDi_SoHmPVYF0mfKepLcXufwB?usp=sharing). You can download them for feature extraction.

## Inference to extract video features
You can extract video features by running:
```
# extract video feats for A1
python inference_video_feature_vitg.py \
    --video_dir $BASE_DIR/data/crop_videos/A1 \
    --ckpt_pth workdir/track3_videomae_ego_verb/Crop_Mix_lr_1e-3_epoch_35/checkpoint-best/mp_rank_00_model_states.pt \
    --output_dir $BASE_DIR/data/extracted_features/A1

# extract video feats for A2
python inference_video_feature_vitg.py \
    --video_dir $BASE_DIR/data/crop_videos/A2 \
    --ckpt_pth workdir/track3_videomae_ego_verb/Crop_Mix_lr_1e-3_epoch_35/checkpoint-best/mp_rank_00_model_states.pt \
    --output_dir $BASE_DIR/data/extracted_features/A2
```

Also we provide extracted feats for dataset A1&A2 here: xxx.
