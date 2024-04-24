# Augmented Self-Mask Attention Transformer for Naturalistic Driving Action Recognition

The 1nd Place Submission to The 8th NVIDIA AI City Challenge (2024) Track 3: Naturalistic Driving Action Recognition.

<!-- ![framework](/assets/framework.png) -->

<div align=center>
<img src="assets/framework.png" width="700px" >
</div>


## Data Structure
```
Aicity2024-track3/data
|--raw_videos/              # original videos
|   |--A1/                  
|       |--user_id_xxx/
|           |--xxx.MP4
|   |--A2/                  
|   |--labels&instructions/ 
|       |--A1/   
|           |--xxx.csv          
|--crop_videos/             # videos of crop human, structure like raw_videos
|   |--A1/
|       |--user_id_xxx/
|           |--xxx.mp4
|   |--A2/
|--splited_videos/          # videos splited by label
|   |--A1/
|       |--xxx.MP4
            ...
|       |--splited_videos_label.csv  
|--label_A1-train_A1-val.json
|--label_submit.json
```

## Workflow

The workflow for training action classification model is as follow:

1. [Dataset Preparation](docs/PREPROCESS.md)
2. [Feature Extraction](docs/FEATURE_EXTRACTION.md)
3. [AMA](docs/AMA.md)
4. [Post Process](docs/POST_PROCESS.md)

## Example for Test on Dataset B
1. [Test Example](docs/TEST.md)


## Contact
If you have any questions, feel free to contact Tiantian Zhang ( zhangtt13@chinatelecom.cn / zhangtiantian_01@163.com ).
