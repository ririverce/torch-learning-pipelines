import copy
import numpy as np
import cv2


class AnchorBoxConverter:

    def __init__(self, num_classes, default_box, image_size,
                 iou_threshold=0.5, enable_augmentation=True):
        self._num_classes = num_classes
        self._default_box = default_box
        self._image_size = image_size
        self._iou_threshold = iou_threshold
        self._enable_augmentation = enable_augmentation

    def _calc_iou_matrix(self, batch_bboxes):
        default_box_cxy = self._default_box[:, :2]
        default_box_wh = self._default_box[:, 2:]
        default_box_tl = default_box_cxy - (default_box_wh / 2.0)
        default_box_br = default_box_cxy + (default_box_wh / 2.0)
        default_box = np.concatenate([default_box_tl, default_box_br], axis=-1)
        iou_matrix_list = []
        for bboxes in batch_bboxes:
            iou_matrix = np.zeros([default_box.shape[0], len(bboxes)])
            for i, box in enumerate(bboxes, 0):
                tiled_box = np.tile([box], [default_box.shape[0], 1])
                tiled_box_tl = tiled_box[:, :2]
                tiled_box_br = tiled_box[:, 2:]
                tiled_box_wh = tiled_box_br - tiled_box_tl
                intersection_tl = np.maximum(default_box_tl, tiled_box_tl)
                intersection_br = np.minimum(default_box_br, tiled_box_br)
                intersection_wh = intersection_br - intersection_tl
                intersection_wh = np.clip(intersection_wh, 0, None)
                intersection_area = intersection_wh[:, 0] * intersection_wh[:, 1]
                default_box_area = default_box_wh[:, 0] * default_box_wh[:, 1]
                tiled_box_area = tiled_box_wh[:, 0] * tiled_box_wh[:, 1]
                union_area = default_box_area + tiled_box_area - intersection_area
                iou_matrix[:, i] = intersection_area / union_area
            iou_matrix_list.append(iou_matrix)
        return iou_matrix_list

    def _calc_matched_pair(self, iou_matrix_list):
        iou_matrix_list = copy.deepcopy(iou_matrix_list)
        matched_pair_list = []
        for iou_matrix in iou_matrix_list:
            matched_pair = []
            for _ in range(iou_matrix.shape[1]):
                max_args = [np.argmax(np.max(iou_matrix, axis=1)),
                            np.argmax(np.max(iou_matrix, axis=0))]
                iou_matrix[max_args[0]] = 0
                iou_matrix[:, max_args[1]] = 0
                matched_pair.append(max_args)
            matched_pair_list.append(matched_pair)
        return matched_pair_list
    
    def _remove_matched_iou(self, iou_matrix_list, matched_pair_list):
        for i, matched_pair in enumerate(matched_pair_list):
            for pair in matched_pair:
                iou_matrix_list[i][pair] = 0.0
        return iou_matrix_list

    def _calc_surrounding_pair(self, iou_matrix_list):
        matched_pair_list = []
        for iou_matrix in iou_matrix_list:
            matched_pair = []
            if 0 in iou_matrix.shape:
                matched_pair_list.append(matched_pair)
                continue
            while np.max(iou_matrix) >= self._iou_threshold:
                max_args = [np.argmax(np.max(iou_matrix, axis=1)),
                            np.argmax(np.max(iou_matrix, axis=0))]
                iou_matrix[max_args[0]] = 0
                matched_pair.append(max_args)
            matched_pair_list.append(matched_pair)
        return matched_pair_list
    
    def _calc_conf(self, batch_labels, matched_pair_list, surrounding_pair_list):
        batch_conf = np.zeros([len(batch_labels),
                               self._default_box.shape[0],
                               self._num_classes])
        batch_conf[:, :, 0] = 1.0
        for i, labels in enumerate(batch_labels, 0):
            for pair in matched_pair_list[i]:
                batch_conf[i, pair[0], labels[pair[1]]] = 1.0
                batch_conf[i, pair[0], 0] = 0.0
            for pair in surrounding_pair_list[i]:
                batch_conf[i, pair[0], labels[pair[1]]] = 1.0
                batch_conf[i, pair[0], 0] = 0.0
        return batch_conf

    def _calc_loc(self, batch_bboxes, matched_pair_list, surrounding_pair_list):
        batch_loc = np.zeros([len(batch_bboxes),
                              self._default_box.shape[0],
                              4])
        for i, bboxes in enumerate(batch_bboxes, 0):
            for pair in matched_pair_list[i]:
                d_cx, d_cy, d_w, d_h = self._default_box[pair[0]]
                b_xmin, b_ymin, b_xmax, b_ymax = bboxes[pair[1]]
                b_cx, b_cy = (b_xmax + b_xmin) / 2, (b_ymax + b_ymin) / 2
                b_w, b_h = b_xmax - b_xmin, b_ymax - b_ymin
                delta_cx = (b_cx - d_cx) / (d_w * 0.1)
                delta_cy = (b_cy - d_cy) / (d_h * 0.1)
                delta_w = np.log(b_w / d_w) / 0.2
                delta_h = np.log(b_h / d_h) / 0.2
                batch_loc[i, pair[0]] = [delta_cx, delta_cy, delta_w, delta_h]
            for pair in surrounding_pair_list[i]:
                d_cx, d_cy, d_w, d_h = self._default_box[pair[0]]
                b_xmin, b_ymin, b_xmax, b_ymax = bboxes[pair[1]]
                b_cx, b_cy = (b_xmax + b_xmin) / 2, (b_ymax + b_ymin) / 2
                b_w, b_h = b_xmax - b_xmin, b_ymax - b_ymin
                delta_cx = (b_cx - d_cx) / (d_w * 0.1)
                delta_cy = (b_cy - d_cy) / (d_h * 0.1)
                delta_w = np.log(b_w / d_w) / 0.2
                delta_h = np.log(b_h / d_h) / 0.2
                batch_loc[i, pair[0]] = [delta_cx, delta_cy, delta_w, delta_h]
        return batch_loc

    def convert(self, batch_data):
        batch_labels = [
            data["class_ids"] for data in batch_data
        ]
        batch_bboxes = [
            [
                [
                    box[0] / data["image"].shape[1],
                    box[1] / data["image"].shape[0],
                    box[2] / data["image"].shape[1],
                    box[3] / data["image"].shape[0],
                ] for box in data["bboxes"]
            ] for data in batch_data
        ]
        batch_inputs = []
        for data in batch_data:
            image = cv2.resize(data["image"], self._image_size)
            batch_inputs.append(image)
        batch_inputs = np.transpose(batch_inputs, (0, 3, 1, 2))
        iou_matrix_list = self._calc_iou_matrix(batch_bboxes)
        matched_pair_list =  self._calc_matched_pair(iou_matrix_list)
        iou_matrix_list = self._remove_matched_iou(iou_matrix_list,
                                                   matched_pair_list)
        surrounding_pair_list = self._calc_surrounding_pair(iou_matrix_list)
        batch_conf = self._calc_conf(batch_labels,
                                     matched_pair_list,
                                     surrounding_pair_list)
        batch_loc = self._calc_loc(batch_bboxes,
                                   matched_pair_list,
                                   surrounding_pair_list)        
        batch_data = {
            "images": [data["image"] for data in batch_data],
            "input" : batch_inputs.astype(np.float32),
            "conf"  : batch_conf.astype(np.float32),
            "loc"   : batch_loc.astype(np.float32)
        }
        return batch_data