import numpy as np
import cv2


def generate_default_box(input_size, num_grids, step, size, aspect_ratio):
    default_box = []
    for l in range(len(num_grids)):
        for y in range(num_grids[l][1]):
            for x in range(num_grids[l][0]):
                cx = (x + 0.5) * step[l][0]
                cy = (y + 0.5) * step[l][1]
                w = size[l][0]
                h = size[l][1]
                default_box.append([cx, cy, w, h])
                w = np.sqrt(size[l][0] * size[l+1][0])
                h = np.sqrt(size[l][1] * size[l+1][1])
                default_box.append([cx, cy, w, h])
                for ar in aspect_ratio[l]:
                    w = size[l][0] * np.sqrt(ar)
                    h = size[l][1] / np.sqrt(ar)
                    default_box.append([cx, cy, w, h])
                    w = size[l][0] / np.sqrt(ar)
                    h = size[l][1] * np.sqrt(ar)
                    default_box.append([cx, cy, w, h])
    default_box = np.array(default_box)
    default_box[:, :2] /= input_size
    default_box[:, 2:] /= input_size
    return default_box


def default_box_ssdwide():
    grids = [
        [50, 29],
        [25, 15],
        [13, 8],
        [7, 4],
        [4, 2],
        [2, 1]
    ]
    steps = [
        [8, 8],
        [16, 15],
        [31, 28],
        [57, 56],
        [100, 113],
        [200, 225]
    ]
    sizes = [
        [30, 30],
        [60, 60],
        [111, 111],
        [162, 162],
        [213, 213],
        [264, 264],
        [315, 315]
    ]
    aspects = [
        [2,],
        [2, 3],
        [2, 3],
        [2, 3],
        [2,],
        [2,],
    ]
    return generate_default_box((320, 180), grids, steps, sizes, aspects)


def draw_voc_bboxes(image, default_box, conf, loc):
    voc_color_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0],
                     [128, 128, 0], [0, 0, 128], [128, 0, 128],
                     [0, 128, 128], [128, 128, 128], [64, 0, 0],
                     [192, 0, 0], [64, 128, 0], [192, 128, 0],
                     [64, 0, 128], [192, 0, 128], [64, 128, 128],
                     [192, 128, 128], [0, 64, 0], [128, 64, 0],
                     [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    default_box_cxy = default_box[:, :2] 
    default_box_wh = default_box[:, 2:]
    loc_cxy = loc[:, :2]
    loc_wh = loc[:, 2:]
    bbox_cxy = loc_cxy * 0.1 * default_box_wh + default_box_cxy
    bbox_wh = np.exp(loc_wh * 0.2) * default_box_wh
    mask = np.max(conf[:, 1:], -1) > 0.5
    labels = np.argmax(conf, -1)[mask]
    bbox_cxy = bbox_cxy[mask]
    bbox_wh = bbox_wh[mask]
    bbox_tl = bbox_cxy - bbox_wh / 2
    bbox_br = bbox_cxy + bbox_wh / 2
    bboxes = np.concatenate([bbox_tl, bbox_br], -1)
    height, width = image.shape[:2]
    for box, label in zip(bboxes, labels):
        x_min = int(box[0] * width)
        y_min = int(box[1] * height)
        x_max = int(box[2] * width)
        y_max = int(box[3] * height)
        color = voc_color_map[label][::-1]
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                              color, 1)
#        cv2.imshow('test', image)
#        cv2.waitKey(1)
    """
    default_box_wh = default_box_wh[mask]
    default_box_cxy = default_box_cxy[mask]
    default_box_tl = default_box_cxy - default_box_wh / 2
    default_box_br = default_box_cxy + default_box_wh / 2
    bboxes = np.concatenate([default_box_tl, default_box_br], -1)
    for box in bboxes:
        x_min = int(box[0] * width)
        y_min = int(box[1] * height)
        x_max = int(box[2] * width)
        y_max = int(box[3] * height)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 1)
    """
    return image
