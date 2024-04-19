#coding:utf-8
import csv
import os
import cv2
import json
import argparse
import glob


shell = {
	"version":"AI CITY 2024 track-3",
	"database":{}
}


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

classes = ['Normal Forward Driving',
           'Drinking', 
           'Phone Call(right)', 
           'Phone Call(left)', 
           'Eating',
           'Text (Right)', 
           'Text (Left)', 
           'Reaching behind', 
           'Adjust control panel', 
           'Pick up from floor (Driver)', 
           'Pick up from floor (Passenger)', 
           'Talk to passenger at the right', 
           'Talk to passenger at backseat', 
           'yawning', 
           'Hand on head', 
           'Singing or dancing with music']

def load_anno_csv(obj_path, video_dict):
    video_list = {}
    with open(obj_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_name = row['Filename'] if ' ' not in row['Filename'] else video_name
            user_id = os.path.basename(obj_path).split('.')[0]
            location = row['Camera View'].replace(' ', '')
            appearance = video_name.split('.')[0].split('_')[-1]
            video_name = video_dict[user_id][location][appearance]
            video_list.setdefault(video_name, {'annotations': []})
            label_id = int(row['Label (Primary)'][6:])
            label = classes[label_id]
            segment = [timestamp_to_seconds(row['Start Time']), timestamp_to_seconds(row['End Time'])]
            if segment[0] == segment[1]:
                segment[1] += 1
            
            if (os.path.basename(video_name).split('.')[0] == 'Right_side_window_user_id_16700_NoAudio_7' and segment[1] > 268) or (os.path.basename(video_name).split('.')[0] == 'Dashboard_user_id_59359_NoAudio_5' and segment[1] > 322):
                continue

            video_list[video_name]['annotations'].append({
                'label_id': label_id,
                'label': label,
                'segment': segment
            })
        
    return video_list



def collect_videos(video_path):
    videos = glob.glob(os.path.join(video_path, '*/*.mp4'))
    video_dict = {}
    for video in videos:
        user_id, video_name = video.split('/')[-2:]
        location = local_dict[video_name.split('_')[0]]
        appearance = video_name.split('.')[0].split('_')[-1]
        
        video_dict.setdefault(user_id, {})
        video_dict[user_id].setdefault(location, {})
        video_dict[user_id][location][appearance] = video
    
    return video_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    
    parser.add_argument('--video_path', metavar='path', required=True,
                        help='the path to the A1 dataset ')   
    parser.add_argument('--label_path', metavar='path', required=True,
                        help='the path to the annotation files of A1')
    parser.add_argument('--output_path', metavar='path', required=True,
                        help='the path to the generated json file')

    args = parser.parse_args()

    train_video_dict = collect_videos(args.video_path)

    for label_file in glob.glob(os.path.join(args.label_path, f'*.csv')):
        user_id = os.path.basename(label_file).split('.')[0]
        video_list = load_anno_csv(label_file, train_video_dict)

        for video_path in video_list:
            cap = cv2.VideoCapture(video_path)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_time = frame_length / fps
            
            video_name = os.path.basename(video_path).split('.')[0]
            video_list[video_path]['resolution'] = f'{frame_w}*{frame_h}'
            video_list[video_path]['duration'] = video_time
            video_list[video_path]['subset'] = 'training' if user_id not in validata_ids else 'validation'
        
            shell['database'][video_name] = video_list[video_path]

    with open(args.output_path, 'w') as fp:
        json.dump(shell, fp, ensure_ascii=False, indent=4)