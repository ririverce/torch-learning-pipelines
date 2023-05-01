import colorsys
import cv2


def int2color(x):
    h = (x * 0.31) % 1.0
    s = 1.0
    v = 1.0 - ((x * 0.13) % 0.5)
    color = colorsys.hsv_to_rgb(h, s, v)
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    return color

def bbox_overlay(data):    
    image = data["image"]
    for label, class_id, bbox in zip(data['labels'], data['class_ids'], data['bboxes']):
        color = int2color(class_id)
        dark_color = (int(color[0]/2), int(color[1]/2), int(color[2]/2))
        image = cv2.rectangle(
            image, 
            (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
            color, 2, 
        )
        dy = len(label) * 6 + 17
        image = cv2.rectangle(
            image, 
            (int(bbox[0]), int(bbox[1])-15), (int(bbox[0])+dy, int(bbox[1])),
            dark_color, -1, 
        )
        image = cv2.rectangle(
            image, 
            (int(bbox[0]), int(bbox[1])-15), (int(bbox[0])+dy, int(bbox[1])),
            color, 2, 
        )
        image = cv2.putText(
            image, label,
            (int(bbox[0])+2, int(bbox[1])-2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=color, thickness=1
        )
    return image