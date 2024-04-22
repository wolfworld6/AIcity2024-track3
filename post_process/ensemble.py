import os
import pandas as pd
import numpy as np
import argparse


def segment_iou_(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[0])
    tt2 = np.minimum(target_segment[1], candidate_segments[1])
    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (
        (candidate_segments[1] - candidate_segments[0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    if segments_union == 0:
        return 0
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def merge_same_lable_by_tiou_and_score(infos, tiou_threshold):
    new_info = {}
    for vid in infos:
        for cat in infos[vid]:
            items = infos[vid][cat]
            num = len(items)
            visited = [0] * num

            for i in range(num - 1):
                if visited[i]:
                    continue
                candidate_segments = [
                    [items[i]["seg"][0], items[i]["seg"][1], items[i]["score"]]
                ]
                for j in range(i + 1, num):
                    if visited[j]:
                        continue
                    segment1 = items[j]["seg"]
                    segment2 = items[i]["seg"]
                    tiou = segment_iou_(items[i]["seg"], items[j]["seg"])
                    if tiou > tiou_threshold:
                        candidate_segments.append(
                            [items[j]["seg"][0], items[j]["seg"][1], items[j]["score"]]
                        )
                        visited[j] = 1
                visited[i] = 1

                if len(candidate_segments) == 1 and candidate_segments[0][2] < 0:
                    continue
                st = 0
                et = 0
                weights = 0
                for seg in candidate_segments:
                    st += seg[0] * seg[2]
                    et += seg[1] * seg[2]
                    weights += seg[2]
                if vid not in new_info:
                    new_info[vid] = {}
                if cat not in new_info[vid]:
                    new_info[vid][cat] = []
                item = {}
                item["seg"] = [int(st / weights), int(et / weights)]
                new_info[vid][cat].append(item)
    return new_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ensemble")

    parser.add_argument(
        "--out_file",
        metavar="path",
        required=False,
        help="the file name of the merged txt file",
    )
    parser.add_argument(
        "--merge_tiou_threshold",
        required=False,
        type=float,
        default=0.01,
        help="the tiou_threshold when merging results of same lable",
    )
    args = parser.parse_args()

    out_txt = args.out_file
    merge_tiou_threshold = args.merge_tiou_threshold

    txt_list = [
        "./submit/maev1_AMA_f32_f16_score0.2add_score.txt",
        "./submit/maev2_AMA_f16_f16_score0.2add_score.txt",
    ]
    infos = {}
    for txt_path in txt_list:
        with open(txt_path, "r") as txt_file:
            for line in txt_file.readlines():
                vid, cat, st, et, score = line.strip().split(" ")
                if vid not in infos:
                    infos[vid] = {}
                if cat not in infos[vid]:
                    infos[vid][cat] = []
                item = {}
                item["seg"] = [int(st), int(et)]
                tmp_s = score.split(".")[1]
                tmp_l = len(tmp_s)
                tmp_score = int(tmp_s) / pow(10, tmp_l)
                item["score"] = tmp_score
                infos[vid][cat].append(item)

    new_info = merge_same_lable_by_tiou_and_score(infos, merge_tiou_threshold)

    n1 = 0
    f1 = open(out_txt, "w+")
    for video_id in range(1, 31):
        for i in range(1, 16):
            if str(i) in new_info[str(video_id)]:
                item = new_info[str(video_id)][str(i)]
                num = len(item)
                for j in range(num):
                    if item[j]["seg"] == ["#"]:
                        continue
                    if item[j]["seg"][1] - item[j]["seg"][0] > 30:
                        continue
                    f1.write(
                        str(video_id)
                        + " "
                        + str(i)
                        + " "
                        + str(item[j]["seg"][0])
                        + " "
                        + str(item[j]["seg"][1])
                        + "\n"
                    )
