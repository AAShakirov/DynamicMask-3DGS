import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
image_origin = cv2.imread("0001.jpg")
h, w = image_origin.shape[:2]

results = model.predict(
    source=image_origin,
    classes=[0, 2],
    conf=0.25
)

object_mask = np.zeros((h, w), dtype=np.uint8)
class_mask = np.zeros((h, w), dtype=np.uint8)

for r in results:
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        
        if r.masks is not None:
            masks = r.masks.xy
            
            for idx, (box, cls, mask) in enumerate(zip(boxes, clss, masks)):
                class_id = int(cls)
                
                if class_id == 0:
                    mask_value = 1
                elif class_id == 2:
                    mask_value = 2
                
                obj_mask = np.zeros((h, w), np.uint8)
                mask_points = mask.astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(obj_mask, [mask_points], -1, 255, cv2.FILLED)
                object_mask = cv2.bitwise_or(object_mask, obj_mask)
                class_mask[obj_mask == 255] = mask_value

image_4ch = np.zeros((h, w, 4), dtype=np.uint8)
image_4ch[:, :, :3] = image_origin
image_4ch[:, :, 3] = object_mask

np.save('image_4_ch.npy', image_4ch)
