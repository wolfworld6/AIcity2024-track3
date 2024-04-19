# Preprocess

## Prepare
```
export BASE_DIR=/xxx/AIcity2024-track3
cd $BASE_DIR/preprocess
```

## Preprocess Train Videos

1. crop the human body of the input videos
```
python yolov5/driver_tracking.py --vid_path $BASE_DIR/data/raw_videos/A1  --out_file $BASE_DIR/data/crop_videos/A1

python yolov5/driver_tracking.py --vid_path $BASE_DIR/data/raw_videos/A2  --out_file $BASE_DIR/data/crop_videos/A2
```

2. generate json files
```
python get_json_for_data_A1.py --video_path $BASE_DIR/data/crop_videos/A1 --label_path $BASE_DIR/data/raw_videos/labels\&instructions/A1 --output_path $BASE_DIR/data/label_A1-train_A1-val.json

python get_json_for_data_A2.py --video_path $BASE_DIR/data/crop_videos/A2 --output_path $BASE_DIR/data/label_submit.json
```

3. trip videos
```
python get_splited_videos.py --data_path $BASE_DIR/data/crop_videos/A1 --save_path $BASE_DIR/data/splited_videos/A1  --csv_output $BASE_DIR/data/splited_videos/splited_videos_label.csv --label_path $BASE_DIR/data/raw_videos/labels&instructions/A1
```

## Example for Test Videos

We recommend placing and treating dataset *__B__* like dataset *__A2__*

1. crop the human body of the input videos
```
python yolov5/driver_tracking.py --vid_path $BASE_DIR/data/raw_videos/B  --out_file $BASE_DIR/data/crop_videos/B
```

2. generate json files
```
python get_json_for_data_B.py --video_path $BASE_DIR/data/crop_videos/B --output_path $BASE_DIR/data/label_submit_B.json
```