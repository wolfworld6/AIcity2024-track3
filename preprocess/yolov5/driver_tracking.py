# -*- coding: utf-8 -*-


import os
import argparse 
import glob 
import cv2
import torch
from utils.torch_utils import select_device
from tqdm import tqdm


def convert(xyxy):
    x1, y1 = xyxy[0], xyxy[1]
    w = int(xyxy[2]) - int(x1)
    h = int(xyxy[3]) - int(y1)
    
    return (x1,y1,w,h)


def init(video_path):
    cap = cv2.VideoCapture(video_path)
    full_rate = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    return full_rate, width, height, vid_length


def compute_IoU(box1, box2, x1y1x2y2=True,
                GIoU=False, DIoU=False, 
                CIoU=False, eps=1e-7):

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = max((min(b1_x2, b2_x2) - max(b1_x1, b2_x1)),0) * \
            max((min(b1_y2, b2_y2) - max(b1_y1, b2_y1)),0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou  # Iou


def crop_save_driver_vid(vid_name,
                         frames,
                         max_xyxy_list,
                         full_rate,
                         out_file=''):
   
    out_file_name = os.path.join(out_file, vid_name+'.mp4')
    _, _, width, height = convert(max_xyxy_list)
    
    print("saving video: "+ vid_name)
    print("width, hight", int(width),', ', int(height) )
    
    output = cv2.VideoWriter(out_file_name, 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            full_rate, 
                            (int(width), int(height)))
    
    # stream = cv2.VideoCapture(vid_path)
    # counter = 0
    # while (1):
    #     ret, frame = stream.read()
    #     if not ret:
    #          break           
    #     frame = (frame[int(max_xyxy_list[1]):int(max_xyxy_list[3]),
    #                    int(max_xyxy_list[0]):int(max_xyxy_list[2])])   
    #     output.write(frame)
    #     counter += 1
    #     if counter% 1000 == 0:
    #         print(counter, "frames has been saved")
    # print("video " , vid_name, " has been saved")
    # return
    for frame in tqdm(frames):
        frame = (frame[int(max_xyxy_list[1]):int(max_xyxy_list[3]), int(max_xyxy_list[0]):int(max_xyxy_list[2])])
        output.write(frame)
    
    return


def driver_tube_construction(vid_name, frame_predictions, vehicles_tubes, iou_threshhold = 0.3):

    for x1y1x2y2 in frame_predictions:
        if x1y1x2y2[5] !=0 or x1y1x2y2[4] <= 0.50: ###class number 
            continue
        
        # flag for appending bbox == new car   False ==> not yet appended  True ==> appended
        append_flag = False

        # if there is pervious detected car check if the car is the same 
        if vehicles_tubes:
            for bbox_num in range(len(vehicles_tubes)):

                iou = compute_IoU(vehicles_tubes[bbox_num]['xyxy_list'][-1], x1y1x2y2)
                if iou >= iou_threshhold:
                 
                    vehicles_tubes[bbox_num]['xyxy_list'].append(x1y1x2y2.tolist())

                    append_flag = True
                    # end -- searching for matching car has been completed

        # append new detected car 
        if not append_flag:
            vehicles_tubes.append({'xyxy_list':[x1y1x2y2.tolist()]})
        
    if len(frame_predictions) != 0 and ((int(frame_predictions[-1][2]) - int(frame_predictions[-1][0]) == 1503 and 
                                        int(frame_predictions[-1][3]) - int(frame_predictions[-1][1]) == 830) or
                                        (int(frame_predictions[-1][2]) - int(frame_predictions[-1][0]) == 1589 and
                                        int(frame_predictions[-1][3]) - int(frame_predictions[-1][1] == 987))):
    
        print("width, hight in exception", frame_predictions[-1][2] - frame_predictions[-1][0],
              frame_predictions[-1][3] - frame_predictions[-1][1])
        print ("vid name in exception",vid_name )
       
        for index in reversed(range(len(vehicles_tubes))):
            # if there is pervious detected car check if the car is the same 
            if vehicles_tubes:
                for bbox_num in range(len(vehicles_tubes)):
                    iou = compute_IoU(vehicles_tubes[bbox_num]['xyxy_list'][-1], x1y1x2y2)
                    if iou >= iou_threshhold:
                        vehicles_tubes[bbox_num]['xyxy_list'].append(x1y1x2y2.tolist())

    return


def main(model, vid_name, video_path, out_file):
    full_rate, width, height, vid_length = init(video_path)
    print(vid_name, full_rate, width, height, vid_length )
    
    stream = cv2.VideoCapture(video_path)

    vehicles_tubes = [] 
    frames = []
    while (1):       
        ret, frame = stream.read()       
        if not ret:
             break
             
        predictions = model(frame)
        for frame_predictions in predictions.xyxy:
            driver_tube_construction(vid_name, frame_predictions, vehicles_tubes, iou_threshhold = 0.3)
        
        frames.append(frame)



    # max_area = 0
    # max_xyxy_list = []
    # car = 0
    # for index in reversed(range(len(vehicles_tubes))):
    #     car = vehicles_tubes.pop(index)
    #     area = []
    #     for i in range(len(car['xyxy_list'])):
    #             x, y, width, hight = convert(car['xyxy_list'][i])
    #             area.append(width * hight)

    #     tmp = max(area)
    #     indexx = area.index(tmp)
    #     xyxy_list = car['xyxy_list'][indexx]
       
    #     if tmp > max_area:
    #           max_area = tmp
    #           max_xyxy_list = xyxy_list
    #           car = car

    max_area = 0
    max_xyxy_list = None
    for tube in vehicles_tubes:
        for xmin, ymin, xmax, ymax, _, _ in tube['xyxy_list']:
            if (xmax - xmin) * (ymax - ymin) > max_area:
                max_area = (xmax - xmin) * (ymax - ymin)
                max_xyxy_list = [xmin, ymin, xmax, ymax]

    # max_xyxy_list = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    # for tube in vehicles_tubes:
    #     for xmin, ymin, xmax, ymax, _, _ in tube['xyxy_list']:
    #         if xmin < max_xyxy_list[0]:
    #             max_xyxy_list[0] = xmin
    #         if ymin < max_xyxy_list[1]:
    #             max_xyxy_list[1] = ymin
    #         if xmax > max_xyxy_list[2]:
    #             max_xyxy_list[2] = xmax
    #         if ymax > max_xyxy_list[3]:
    #             max_xyxy_list[3] = ymax

    if len(vehicles_tubes): 
        result = crop_save_driver_vid(vid_name, frames, max_xyxy_list, full_rate, out_file)
    
    print(f'video {vid_name} has been saved !!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='driver tracking')
    parser.add_argument('--vid_path', metavar='path', required=False, help='the path to the folder that contains video/videos in .mp4 format .. ')
    parser.add_argument('--out_path', metavar='path', required=False, help='the path where the tracked driver videos will be saved ..')
    parser.add_argument('--device', required=False, type=int, default=0)
    args = parser.parse_args()

    videos_path = glob.glob(f'{args.vid_path}/*/*.MP4')
    # out_path = f'./device_{args.device}'
    # videos_path = ['/data01/qtwang/Aicity2023-Track3/data/A1_1/user_id_13522/Dashboard_user_id_13522_NoAudio_5.MP4']
    
    
    model = torch.hub.load('yolov5','yolov5s', pretrained=True, source='local', force_reload=False, device=f'cuda:{args.device}').autoshape()
    # device = select_device(args.device)
    # model = model.to(device)
    # model = model.to(f'cuda:{args.device}')
        
    for video_path in videos_path:
        if video_path[-4:]==".MP4":
            vid_name = str(os.path.basename(video_path))[:-4]
            print("start processing video: ", vid_name)
            tmp_path = "user_id_" + vid_name.split("_")[-3]
            # out_path = out_path + tmp_path
            out_path = os.path.join(args.out_path, tmp_path)
            print("out_path",out_path)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            main(model, vid_name, video_path, out_path)