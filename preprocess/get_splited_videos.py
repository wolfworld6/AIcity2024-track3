import subprocess
import os
import csv
import argparse
import glob
from tqdm import tqdm

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


def load_anno_csv(obj_path):
    video_list = {}
    with open(obj_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_name = row['Filename'] if ' ' not in row['Filename'] else video_name
            video_list.setdefault(video_name, [])
            video_list[video_name].append({
                'label': row['Label (Primary)'][6:],
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
    with open(args.csv_output, "w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["video_name","label"])

        for label_file in tqdm(glob.glob(os.path.join(args.label_path, f'*.csv'))):
            user_id = os.path.basename(label_file).split('.')[0]
            annotations = load_anno_csv(label_file)
            for video_name, annotations in load_anno_csv(label_file).items():
                for idx, ann in enumerate(annotations):
                    try:
                        video = video_dict[user_id][ann['camera_view']][ann['appearance']]
                    except:
                        import ipdb;ipdb.set_trace()
                    if not os.path.exists(video):
                        raise ValueError(f'No such video: {video}')
                    splited_video_name = f"{ann['camera_view']}_{user_id}_NoAudio_{ann['appearance']}_split_{idx+1}.MP4"
                    writer.writerow([splited_video_name, ann['label']])
                
                    command = f"ffmpeg -i {video} -ss {ann['start_time']} -to {ann['end_time']} -vcodec copy -acodec copy {os.path.join(args.save_path, splited_video_name)}"
                    commands.append(command)
    
    for command in commands:
        os.system(command)

    print('Done !!!')