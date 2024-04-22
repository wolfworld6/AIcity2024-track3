# Preprocess

## Prepare
```
export BASE_DIR=/xxx/AIcity2024-track3
cd $BASE_DIR/preprocess
```

## Preprocess Train Videos

1. Detect driver spatial location in the video, then crop each video based on the driver bounding box.
```
python yolov5/driver_tracking.py --vid_path $BASE_DIR/data/raw_videos/A1  --out_file $BASE_DIR/data/crop_videos/A1

python yolov5/driver_tracking.py --vid_path $BASE_DIR/data/raw_videos/A2  --out_file $BASE_DIR/data/crop_videos/A2
```

2. Prepare json files for the training and validation sets.
```
python get_json_for_data_A1.py --video_path $BASE_DIR/data/crop_videos/A1 --label_path $BASE_DIR/data/raw_videos/labels\&instructions/A1 --output_path $BASE_DIR/data/label_A1-train_A1-val.json

python get_json_for_data_A2.py --video_path $BASE_DIR/data/crop_videos/A2 --output_path $BASE_DIR/data/label_submit.json
```

3. The input videos should be a trimmed videos i.e., contains only one action in each video.
```
python get_splited_videos.py --video_path $BASE_DIR/data/crop_videos/A1 --save_path $BASE_DIR/data/splited_videos/A1  --csv_output $BASE_DIR/data/splited_videos --label_path $BASE_DIR/data/raw_videos/labels&instructions/A1
```