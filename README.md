# AIcity2024-track3


## post-process

### 1. generate txt from csv
```
cd post_process
```

The final best score:
```
python generate_txt.py --csv_path act_pre_ego_crop_AMA_mae2_f16_f16_8h_9k_submmit.csv --out_file submit/act_pre_ego_crop_AMA_mae2_f16_f16_8h_9k_score0.2.txt
```

To be ensembled：
```
python generate_txt.py --csv_path action_predict_info_singleview_crop_AMA_f16_f16_submmit.csv --out_file submit/maev2_AMA_f16_f16_score0.2.txt
```
```
python generate_txt.py --csv_path action_predict_info_singleview_crop_AMA_f32_f16_submmit.csv --out_file submit/maev1_AMA_f32_f16_score0.2.txt
```


### 2. ensemble
```
python ensemble.py --out_file AMA_mergev1_v2.txt
```
