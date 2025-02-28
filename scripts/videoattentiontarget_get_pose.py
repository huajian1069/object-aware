import argparse
import os
import random
from pathlib import Path

import cv2
import pandas as pd
import tqdm
#from retinaface.pre_trained_models import get_model

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

    
# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

random.seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="Root directory of dataset")
    parser.add_argument(
        "--subset", help="Subset of dataset to process", choices=["train", "test"], required=True
    )
    args = parser.parse_args()

    labels_path = f"{args.subset}_release.csv"

    column_names = ["id_col", "path", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "gaze_x", "gaze_y", "fname", "id", "inout",
                    "show-clip", "split", "image_size", "norm_bbox_x_min", "norm_bbox_x_max", "norm_bbox_y_min", "norm_bbox_y_max", "norm_gaze_x", "norm_gaze_y"]
    df = pd.read_csv(
        os.path.join(args.dataset_dir, labels_path),
        sep=",",
        names=column_names,
        usecols=column_names,
        index_col=False,
    )
    df = df.groupby("path")

    #model = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
    #!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True, 
        num_poses=5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5)
    detector = vision.PoseLandmarker.create_from_options(options)

    #detector.eval()

    paths = list(df.groups.keys())

    csv = []
    for path in tqdm.tqdm(paths):
        folder = Path(os.path.dirname(path).split("/")[-1])

        #img = cv2.imread(os.path.join(args.dataset_dir, "images", path))

        image = Image.open(os.path.join(args.dataset_dir, "images", path)).convert("RGB")
        image_np = np.array(image)
        detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np))

        joints = []
        for p in range(len(detection_result.pose_landmarks)):
            f = []
            for j in range(len(detection_result.pose_landmarks[p])):
                f.append(
                (detection_result.pose_landmarks[p][j].x,
                detection_result.pose_landmarks[p][j].y,
                detection_result.pose_landmarks[p][j].z,
                detection_result.pose_landmarks[p][j].visibility,
                detection_result.pose_landmarks[p][j].presence))
            joints.append(f)

        csv.append(
            [
                path,
                joints
            ]
        )

    # Write csv
    df = pd.DataFrame(csv, columns=["path", "joints"])
    df.to_csv(os.path.join(args.dataset_dir, f"{args.subset}_poses.csv"), index=False)
