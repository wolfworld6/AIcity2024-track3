import subprocess
import os
import csv
import argparse
import glob
from tqdm import tqdm
import shutil

_MINUTES_TO_SECONDS=60
_HOURS_TO_SECONDS=3600
def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * _HOURS_TO_SECONDS + minutes * _MINUTES_TO_SECONDS + seconds
    return total_seconds

local_dict = {
    'Dashboard': 'Dashboard',
    'Rear': 'Rearview',
    'Rearview': 'Rearview',
    'Right': 'RightsideWindow'
}

validata_ids = ['user_id_93491', 'user_id_93542', 'user_id_96269', 'user_id_96371', 'user_id_98067',
                'user_id_98389', 'user_id_99635', 'user_id_99660', 'user_id_99882']

def load_anno_csv(obj_path):
    video_list = {}
    with open(obj_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_name = row['Filename'] if ' ' not in row['Filename'] else video_name
            video_list.setdefault(video_name, [])
            video_list[video_name].append({
                'label': row['Label (Primary)'][6:].strip(),
                'start_time': row['Start Time'],
                'end_time': row['End Time'],
                'appearance': video_name.split('_')[-1],
                'camera_view': row['Camera View'].replace(' ', '')
            })
    return video_list


def collect_videos(video_path):
    videos = glob.glob(os.path.join(video_path, '*/*.mp4'))
    video_dict = {}
    for video in videos:
        user_id, video_name = video.split('/')[-2:]
        user_id = user_id.split('.')[0]
        location = local_dict[video_name.split('_')[0]]
        appearance = video_name.split('.')[0].split('_')[-1]
        
        video_dict.setdefault(user_id, {})
        video_dict[user_id].setdefault(location, {})
        video_dict[user_id][location][appearance] = video
    
    return video_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--video_path', metavar='path', required=True, help='the path to the dataset ')   
    parser.add_argument('--label_path', metavar='path', required=True, help='the path to the label')
    parser.add_argument('--save_path', metavar='path', required=True, help='the path to the saved cropped videos')
    parser.add_argument('--csv_output', metavar='path', required=True, help='the path to the generated csv file which contain ("video_name","label")')
    args = parser.parse_args()

    video_dict = collect_videos(args.video_path)

    commands = []
    train_csv = os.path.join(args.csv_output, 'train.csv')
    val_csv = os.path.join(args.csv_output, 'val.csv')
    with open(train_csv, 'w') as trn_fp, open(val_csv, 'w') as val_fp: 
        for label_file in tqdm(glob.glob(os.path.join(args.label_path, f'*.csv'))):
            user_id = os.path.basename(label_file).split('.')[0]
            annotations = load_anno_csv(label_file)
            fp = val_fp if user_id in validata_ids else trn_fp
            for video_name, annotations in load_anno_csv(label_file).items():
                for idx, ann in enumerate(annotations):
                    video = video_dict[user_id][ann['camera_view']][ann['appearance']]
                    if not os.path.exists(video):
                        raise ValueError(f'No such video: {video}')
                    splited_video_name = f"{ann['camera_view']}_{user_id}_NoAudio_{ann['appearance']}_split_{idx+1}.MP4"

                    if (user_id == 'user_id_20090' and ann['appearance'] == '5' and ann['label'] == '1') or \
                    (ann['camera_view'] == 'RightsideWindow' and user_id == 'user_id_16700' and ann['appearance'] == '7' and ann['label'] in ['0', '1', '2', '3', '5', '13', '15']) or \
                    (ann['camera_view'] == 'Dashboard' and user_id == 'user_id_59359' and ann['appearance'] == '5' and ann['label'] in ['0', '1', '4', '3', '9', '11']):
                        continue

                    fp.write(f"{os.path.join(args.save_path, splited_video_name)} {ann['label']}\n")
                
                    command = f"ffmpeg -i {video} -ss {ann['start_time']} -to {ann['end_time']} -vcodec copy -acodec copy {os.path.join(args.save_path, splited_video_name)}"
                    commands.append(command)
    
    for command in commands:
        os.system(command)

    shutil.copy(val_csv, os.path.join(args.csv_output, 'test.csv'))

    print('Done !!!')