import os
import sys
from typing import Tuple
from argparse import ArgumentParser

import cv2
import numpy as np
from ultralytics import YOLO


def get_images_names(dataset_dir_path: str) -> list:
    image_paths = []
    for filename in os.listdir(dataset_dir_path):
        if filename.endswith(('png', 'jpg', 'jpeg')):
            image_paths.append(dataset_dir_path + '/' + filename)
    
    return image_paths

def get_segmentator_predicts(model: YOLO, image_path: str) -> Tuple[list, int, int]:
    image_origin = cv2.imread(image_path)
    h, w = image_origin.shape[:2]
    
    results = model.predict(
        source=image_origin,
        classes=[0, 1, 2],        # person, bycucle, car
        conf=0.25
    )
    return results, h, w

def get_mask(results: list, h: int, w: int):
    object_mask = np.full((h, w), 255, dtype=np.uint8)
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:            
            if r.masks is not None:
                masks = r.masks.xy
                
                for mask in masks:
                    mask_points = mask.astype(np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(object_mask, [mask_points], 0)

    return object_mask

def segmentate(dataset_dir_path: str, save_png: bool = False):
    image_paths = get_images_names(dataset_dir_path)
    model = YOLO("yolo11n-seg.pt")

    for image_path in image_paths:
        if 'mask' in image_path:
            print(f'{image_path} already exisit! Continue')
            continue
        print(image_path)
        results, h, w = get_segmentator_predicts(model, image_path)
        object_mask = get_mask(results, h, w)
        np.save(f'{image_path.split('.')[0]}_mask.npy', object_mask)
        if save_png:
            cv2.imwrite(f'{image_path.split('.')[0]}_mask.png', object_mask)

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--dataset_dir_path', type=str, default="/images")
    parser.add_argument('--save_png', type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    segmentate(args.dataset_dir_path, args.save_png) # set save_png=True for debug
