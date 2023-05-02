import os
import json
import random
import copy


class BDD100KDetectionLoader:

    _TRAIN_IMAGE_DIR = "images/100k/train"
    _VAL_IMAGE_DIR = "images/100k/val"
    _TRAIN_LABEL_PATH = "labels/bdd100k_labels_images_train.json"
    _VAL_LABEL_PATH = "labels/bdd100k_labels_images_val.json"
    _LABEL_TO_CLASS_ID = {
        'car' : 1,
        'truck' : 2,
        'bus' : 3,
        'motor' : 4,
        'bike' : 5,
        'person' : 6,
        'rider' : 7,
        'traffic sign' : 8,
        'traffic light' : 9,
        'train' : 10
    }

    def __init__(self, root_dir, seed=0):
        self._root_dir = root_dir
        self._seed = seed
        # prepare train data
        image_dir = os.path.join(self._root_dir, self._TRAIN_IMAGE_DIR)
        label_path = os.path.join(self._root_dir, self._TRAIN_LABEL_PATH)
        #image_dir = os.path.join(self._root_dir, self._VAL_IMAGE_DIR)
        #label_path = os.path.join(self._root_dir, self._VAL_LABEL_PATH)
        with open(label_path, "r") as f:
            meta_list = json.load(f)
        dataset = []
        for meta in meta_list:
            image_path = os.path.join(image_dir, meta["name"])
            labels = []
            class_ids = []
            bboxes = []
            for l in meta["labels"]:
                if "box2d" not in l.keys():
                    continue
                labels.append(l["category"])
                class_ids.append(self._LABEL_TO_CLASS_ID[l["category"]])
                bboxes.append(
                    [
                        l['box2d']['x1'],
                        l['box2d']['y1'],
                        l['box2d']['x2'],
                        l['box2d']['y2']
                    ]
                )
            dataset.append(
                {
                    "image_path": image_path,
                    "labels": labels,
                    "class_ids": class_ids,
                    "bboxes": bboxes,
                }
            )
        self._dataset = dataset 
        random.seed(self._seed)
        random.shuffle(self._dataset)
        self._train_dataset = dataset[:int(len(dataset)*0.8)]
        self._val_dataset = dataset[int(len(dataset)*0.8):]

    def get_train_dataset(self):
        return copy.deepcopy(self._train_dataset)

    def get_val_dataset(self):
        return copy.deepcopy(self._val_dataset)