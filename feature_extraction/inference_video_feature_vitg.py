import os 

import cv2
from typing import List

import torch
import torch.nn.functional as F
import numpy as np

import argparse
from models.modeling_feature import vit_giant_patch14_224
import utils
import glob

from tqdm import tqdm

import argparse
import re

def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Inference For Feature Extraction")
    parser.add_argument("--device", type=str, default="cuda:0", help="device select to run function")
    parser.add_argument("--video_dir", type=str, default="./data/crop_videos/A1", help="folder path of input videos")
    parser.add_argument("--ckpt_pth", type=str, default="./workdir/track3_videomae_ego_verb/Crop_Mix_lr_1e-3_epoch_35/checkpoint-best/mp_rank_00_model_states.pt", help="path of model checkpoint used to get feature")
    parser.add_argument("--output_dir", type=str, default='./data/extracted_features/v2_ego4d_verb_frame-16_sr-1/A1', help="path to save extracted video features")
    args = parser.parse_args()


    model = vit_giant_patch14_224()
    checkpoint = torch.load(args.ckpt_pth, map_location='cpu')

    utils.load_state_dict(model, checkpoint['module'], prefix='')

    model.to(args.device)
    model.eval()

    video_list = sorted(glob.glob(os.path.join(args.video_dir, f'*/*.mp4')))
    for video_path in video_list:
        # if re.search(r"user_id_(\d\d)", video_path).group(1) not in args.user_id:
            # continue
        print(video_path)
    
        cap = cv2.VideoCapture(video_path)
        len_frames = int(cap.get(7))
        print(len_frames//16)
        imgs_bs = []
        feat_arr,v_arr,n_arr,a_arr = None,None,None,None
        for i in range(len_frames):
            success, frame = cap.read()
            if not success: continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs_bs.append(img_rgb)


            if (i+1)%16 ==0 and i>20:
                imgs_bs_torch = torch.as_tensor(np.stack(imgs_bs[::2])) #32帧抽取16帧

                # for videomae
                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                frames = imgs_bs_torch # T H W C
                frames = tensor_normalize(frames, mean, std)
            
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                imgs_bs = imgs_bs[(len(imgs_bs)-16):]

                with torch.no_grad():
                    video_inputs = []

                    frames = torch.nn.functional.interpolate(
                        frames,
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    video_input = frames#

                    video_input = torch.tensor(np.array(video_input)).cuda(args.device)
                    video_input = torch.unsqueeze(video_input,dim=0).cuda(args.device)
                    feature,prediction = model(video_input) #3,3806  ,3 × 1024
                    feature = feature.flatten()
                    prediction = F.softmax(prediction, dim=1) # #3,3806

                    feat   = feature.detach().cpu().numpy()[None,...]
                    a_prob = prediction.detach().cpu().numpy()

                    if feat_arr is None:
                        feat_arr = feat
                        a_arr = a_prob
                    else:
                        feat_arr = np.concatenate((feat_arr, feat), axis=0)
                        a_arr    = np.concatenate((a_arr, a_prob), axis=0)

        print(feat_arr.shape)
        print(a_arr.shape)
        print(len_frames//32)

        out_path = os.path.join(args.output_dir, video_path.split('/')[-2])
        os.makedirs(out_path, exist_ok=True)
        out_file = video_path.split('/')[-1].split('.')[0] + '.npz'
        print( os.path.join(out_path, out_file))
        np.savez( os.path.join(out_path, out_file), feats=feat_arr, a_prob=a_arr)
        print(" extractor video Done.")
    
        cap.release()