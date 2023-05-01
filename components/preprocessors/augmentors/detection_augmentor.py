import random
import copy

import numpy as np
import cv2


def bbox_random_crop(data, min_crop_size=(300, 300),
                     keep_aspect=True, threshold=0.1, prob=1.0):
    image = data["image"]
    if random.random() > prob:
        return data
    src_height, src_width = image.shape[:2]
    min_crop_width, min_crop_height = min_crop_size
    if src_width <= min_crop_width or src_height <= min_crop_height:
        return data
    if keep_aspect:
        aspect_ratio = src_height / src_width
        min_crop_height = int(min_crop_width * aspect_ratio)
        crop_x_min = random.randint(0, src_width - min_crop_width - 1)
        crop_y_min = random.randint(0, src_height - min_crop_height - 1)       
        crop_x_max = random.randint(
            crop_x_min + min_crop_width,
            min(
                src_width,
                crop_x_min + int((src_height - crop_y_min) / aspect_ratio)
            )
        )
        crop_y_max = crop_y_min + int((crop_x_max - crop_x_min) * aspect_ratio)
    else: 
        crop_x_min = random.randint(0, src_width - min_crop_width - 1)
        crop_y_min = random.randint(0, src_height - min_crop_height - 1)
        crop_x_max = random.randint(crop_x_min + min_crop_width, src_width)
        crop_y_max = random.randint(crop_y_min + min_crop_height, src_height)
    image = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    crop_width = crop_x_max - crop_x_min
    crop_height = crop_y_max - crop_y_min
    crop_labels = []
    crop_class_ids = []
    crop_bboxes = []
    for label, class_id, bbox in zip(data["labels"], data["class_ids"], data["bboxes"]):
        src_box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        x_min = np.clip(0, bbox[0] - crop_x_min, crop_width)
        y_min = np.clip(0, bbox[1] - crop_y_min, crop_height)
        x_max = np.clip(0, bbox[2] - crop_x_min, crop_width)
        y_max = np.clip(0, bbox[3] - crop_y_min, crop_height)
        if (x_max - x_min) * (y_max - y_min) <= threshold * src_box_area:
            continue
        crop_labels.append(label)
        crop_class_ids.append(class_id)
        crop_bboxes.append([x_min, y_min, x_max, y_max])
    data["image"] = image
    data["labels"] = crop_labels
    data["class_ids"] = crop_class_ids
    data["bboxes"] = crop_bboxes
    return data 