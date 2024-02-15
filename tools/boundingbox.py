import cv2
import numpy as np
import sys

class BoundingBoxUtils:
    @staticmethod
    def draw_bounding_box(img, bounding_boxes, labels=None, normalized=False):
        SCALE = 1
        
        img = cv2.resize(img, (0,0), fx=SCALE, fy=SCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for i, (cx, cy, w, h) in enumerate(bounding_boxes):
            if (normalized):
                cx, w = cx * img.shape[0] / SCALE, w * img.shape[0] / SCALE
                cy, h = cy * img.shape[1] / SCALE, h * img.shape[1] / SCALE
                
            color = BoundingBoxUtils.__box_color(labels[i])

            img = cv2.rectangle(
                img,
                (int(cy - h / 2) * SCALE, int(cx - w / 2) * SCALE),
                (int(cy + h / 2) * SCALE, int(cx + w / 2) * SCALE),
                color,
                1
            )

            if not labels == None:
                img = cv2.putText(
                    img,
                    str(labels[i]),
                    (int(cy - h / 2) * SCALE, int(cx - w / 2) * SCALE - 8),
                    0,
                    0.5,
                    color,
                    1
                )

        return img
    
    @staticmethod
    def __box_color(t):
        t = hash(t) / sys.maxsize

        a = np.array([0.5, 0.5, 0.5], dtype="float32")
        b = np.array([0.5, 0.5, 0.5], dtype="float32")
        c = np.array([1.0, 1.0, 1.0], dtype="float32")
        
        d = np.array([0.263, 0.416, 0.557], dtype="float32")
        
        return (a + b * np.cos(6.28318 * (c * t + d))).tolist()
    
    @staticmethod
    def normalize(bounding_boxes, max_dims):
        bounding_boxes = np.array(bounding_boxes)
        cx = bounding_boxes[:,0] / max_dims[0]
        cy = bounding_boxes[:,1] / max_dims[1]
        w = bounding_boxes[:,2] / max_dims[0]
        h = bounding_boxes[:,3] / max_dims[1]
        return np.array([cx, cy, w, h]).transpose()

    @staticmethod
    def iou(box1, box2, format1="(cx,cy,w,h)", format2="(cx,cy,w,h)"):

        b1x1, b1x2, b1y1, b1y2 = 0, 0, 0, 0
        
        if (format1 == "(cx,cy,w,h)"):
            (cx, cy, w, h) = box1
            b1x1 = cx - w / 2
            b1x2 = cx + w / 2
            b1y1 = cy - h / 2
            b1y2 = cy + h / 2

        b2x1, b2x2, b2y1, b2y2 = 0, 0, 0, 0

        if (format2 == "(cx,cy,w,h)"):
            (cx, cy, w, h) = box2
            b2x1 = cx - w / 2
            b2x2 = cx + w / 2
            b2y1 = cy - h / 2
            b2y2 = cy + h / 2  

        i_x1 = max(b1x1, b2x1)
        i_y1 = max(b1y1, b2y1)
        i_x2 = min(b1x2, b2x2)
        i_y2 = min(b1y2, b2y2)

        def box_area(box):
            (x1, y1, x2, y2) = box
            return abs(x2 - x1) * abs(y2 - y1)

        i_area = box_area(i_x1, i_y1, i_x2, i_y2)
        u_area = box_area(b1x1, b1y1, b1x2, b1y2) + BoundingBoxUtils.box_area(b2x1, b2y1, b2x2, b2y2) - i_area

        return i_area / u_area