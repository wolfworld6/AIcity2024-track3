import os
import pandas as pd
from datetime import *
import csv
import numpy as np
import argparse


video_id_path = "./2024-data_video_ids.csv"
videoNames2Id = {}
with open(video_id_path, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        video_id = int(row["video_id"])  # 假设video_id是整数类型

        for idx in range(1, 4):
            video_file = row[f"video_files{idx}"].replace("'", "")
            videoNames2Id[video_file] = video_id


def get_dict_from_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        database_items = {}
        reader = csv.DictReader(csvfile)
        for row in reader:
            ts = int(row["t-start"].split(".")[0])
            te = int(row["t-end"].split(".")[0])
            # 将时间四舍五入取整
            if row["t-start"].split(".")[1][0] >= "5":
                ts += 1
            if row["t-end"].split(".")[1][0] >= "5":
                te += 1
            id = videoNames2Id[row["video-id"] + ".MP4"]
            label = row["label"]
            if te - ts < 1:
                continue
            if id not in database_items:
                database_items[id] = {}
            if label not in database_items[id]:
                database_items[id][label] = []

            item = {}
            item["seg"] = [ts, te]
            item["score"] = row["score"]
            item["angle"] = row["video-id"].split("_")[0]
            # print(item['angle'])
            database_items[id][label].append(item)
        return database_items


def segment_iou_(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[0])
    tt2 = np.minimum(target_segment[1], candidate_segments[1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (
        (candidate_segments[1] - candidate_segments[0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    if segments_union == 0:
        return 0
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def merge_same_lable_by_tiou(infos, tiou_threshold):
    n = 0
    for vid in infos:
        for cat in infos[vid]:
            items = infos[vid][cat]
            num = len(items)
            for i in range(num - 1):
                # 过滤lable0
                # if cat=="0":
                #     items[i]['score']="#"
                if items[i]["score"] == "#":
                    continue
                for j in range(i + 1, num):
                    if items[j]["score"] == "#":
                        continue
                    if items[i]["score"] == "#":
                        continue
                    target_segment = items[j]["seg"]
                    candidate_segments = items[i]["seg"]
                    tiou = segment_iou_(target_segment, candidate_segments)
                    if tiou > tiou_threshold:
                        n += 1
                        if int(items[i]["score"].split(".")[1][0:3]) >= int(
                            items[j]["score"].split(".")[1][0:3]
                        ):
                            items[j]["score"] = "#"
                        else:
                            items[i]["score"] = "#"

    print("merge num of same lable:", n)


def get_top_1_per_lable(infos):
    n = 0
    scores = {}
    for vid in infos:
        scores[vid] = {}
        for cat1 in infos[vid]:
            items1 = infos[vid][cat1]
            if cat1 not in scores[vid]:
                scores[vid][cat1] = []
            num1 = len(items1)
            for i in range(num1):
                if items1[i]["score"] == "#":
                    continue
                scores[vid][cat1].append(int(items1[i]["score"].split(".")[1][0:3]))

    standard = {}
    for vid in scores:
        standard[vid] = {}
        for cat in scores[vid]:
            v_scroe = scores[vid][cat]
            if len(v_scroe) <= 1:
                continue
            else:
                v_scroe.sort(reverse=True)
                standard[vid][cat] = v_scroe[0]

    for vid in infos:
        for cat1 in infos[vid]:
            items1 = infos[vid][cat1]
            num1 = len(items1)
            for i in range(num1):
                if items1[i]["score"] == "#":
                    continue
                if (
                    cat1 in standard[vid]
                    and int(items1[i]["score"].split(".")[1][0:3]) < standard[vid][cat1]
                ):
                    items1[i]["score"] = "#"
                    n += 1

    print("delete num for top_1 for lable:", n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="time correction for one model result")
    parser.add_argument(
        "--csv_path", metavar="path", required=True, help="the path to the csv file "
    )
    parser.add_argument(
        "--out_file",
        metavar="path",
        required=True,
        help="the file name of the generated txt file",
    )

    parser.add_argument(
        "--merge_tiou_threshold",
        required=False,
        type=float,
        default=0.1,
        help="the tiou_threshold when merging results of same lable",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    out_file = args.out_file
    merge_tiou_threshold = args.merge_tiou_threshold

    # 是否在输出的txt中加入score信息
    # add_score = False
    add_score = True
    if add_score:
        out_path = out_file[:-4] + "add_score.txt"
    else:
        out_path = out_file
    f1 = open(out_path, "w+")

    infos = get_dict_from_csv(csv_path)

    get_top_1_per_lable(infos)
    merge_same_lable_by_tiou(infos, merge_tiou_threshold)

    n1 = 0
    n2 = 0
    id_nums = [0] * 30
    for video_id in range(1, 31):
        # 去掉标签0
        for i in range(1, 16):
            print(infos.keys())
            if str(i) in infos[(video_id)]:
                item = infos[(video_id)][str(i)]
                num = len(item)
                for j in range(num):
                    if item[j]["score"] == "#":
                        n1 += 1
                        continue
                    if int(item[j]["score"].split(".")[1][0:2]) < 20:
                        n2 += 1
                        continue

                    if add_score:
                        f1.write(
                            str(video_id)
                            + " "
                            + str(i)
                            + " "
                            + str(item[j]["seg"][0])
                            + " "
                            + str(item[j]["seg"][1])
                            + " "
                            + item[j]["score"]
                            + "\n"
                        )
                    else:
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

                    id_nums[video_id - 1] += 1
    print("filter num by lable", n1)
    print("filter num by score", n2)
    print("nums of 30 videos", id_nums)
