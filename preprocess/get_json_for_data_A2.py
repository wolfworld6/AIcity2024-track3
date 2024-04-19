import glob
import cv2
import os
import argparse
import json

shell = {
	"version":"AI CITY 2024 track-3",
	"database":{}
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess')
 
    parser.add_argument('--video_path', metavar='path', required=True,
                        help='the path to the A2 dataset ') 

    parser.add_argument('--output_path', metavar='path', required=True,
                        help='the path to the generated json file')
    
    args = parser.parse_args()

    videos = glob.glob(os.path.join(args.video_path, '*/*.mp4'))
    for video_path in videos:
        video_name = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_time = frame_length / fps

        shell['database'][video_name] = {
            'annotations': [],
            'resolution': f'{frame_w}*{frame_h}',
            'duration': video_time,
            'subset': 'validation'
        }

    with open(args.output_path, 'w') as fp:
        json.dump(shell, fp, ensure_ascii=False, indent=4)