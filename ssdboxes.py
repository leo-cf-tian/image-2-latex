import math
import tensorflow as tf

class SSDBoxes:
    '''
    SSD box generator and data transformer
    
    Implementation based on
    - https://arxiv.org/pdf/1512.02325.pdf
    - https://usmanr149.github.io/urmlblog/computer%20vision/2022/09/10/Implenting-SSD-TF2.html
    - https://github.com/TanyaChutani/SSD-Tensorflow2.0/blob/master/notebook/SSD_SKU110K.ipynb
    '''

    @staticmethod
    def default_box_scale(m, k, s_min = 0.2, s_max = 0.9):
        '''
        m feature maps
        s_min lowest layer scale
        s_max highest layer scale

        s_k = s_min + (s_max - s_min) / (m - 1) * (k - 1)
        '''
        return s_min + (s_max - s_min) / (m - 1) * (k - 1)
    
    @staticmethod
    def create_default_boxes(feature_map_shapes=[38, 19, 10, 5, 3, 1], box_aspect_ratios=[3,2,1,1/2,1/3]):

        default_boxes = []

        for k, f_k in enumerate(feature_map_shapes):

            # default box dimensions
            s_k = SSDBoxes.default_box_scale(len(feature_map_shapes), k + 1)
            s_k_prime = math.sqrt(s_k * SSDBoxes.default_box_scale(len(feature_map_shapes), k + 2))\
            
            default_box_shapes = [(s_k * math.sqrt(ar), s_k / math.sqrt(ar)) for ar in box_aspect_ratios] + [(s_k_prime, s_k_prime)]

            for i in range(f_k):
                for j in range(f_k):
                    cx = (i + 0.5) / f_k
                    cy = (j + 0.5) / f_k

                    default_boxes += [(cx, cy, w, h) for (w, h) in default_box_shapes]

        default_boxes = tf.convert_to_tensor(default_boxes, dtype=tf.float32)

        return default_boxes
    
    @staticmethod
    def convert_to_box_form(centre_form):
        centre_form = tf.convert_to_tensor(centre_form)
        box_coordinates = tf.concat([centre_form[:, :2] - centre_form[:, 2:] / 2, 
                                    centre_form[:, :2] + centre_form[:, 2:] / 2 ], 
                                    axis = 1)
        return box_coordinates

    @staticmethod
    def ground_truth_default_iou(gt, db):
        gt = tf.cast(gt, dtype=tf.float32)
        db = tf.cast(db, dtype=tf.float32)

        i_x1 = tf.math.maximum(gt[:, 0], db[:, None, 0])
        i_y1 = tf.math.maximum(gt[:, 1], db[:, None, 1])
        i_x2 = tf.math.minimum(gt[:, 2], db[:, None, 2])
        i_y2 = tf.math.minimum(gt[:, 3], db[:, None, 3])

        i_area = tf.math.maximum(0.0, i_x2 - i_x1) * tf.math.maximum(0.0, i_y2 - i_y1)

        gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        db_area = (db[:, None, 2] - db[:, None, 0]) * (db[:, None, 3] - db[:, None, 1])

        u_area = db_area + gt_area - i_area

        iou = tf.math.maximum(tf.math.divide_no_nan(i_area, u_area), 0)

        return tf.clip_by_value(iou, 0.0, 1.0)
    
    @staticmethod
    def match_to_default_boxes(ground_truth, labels, default_boxes):
        iou_matrix = SSDBoxes.ground_truth_default_iou(
            SSDBoxes.convert_to_box_form(ground_truth),
            SSDBoxes.convert_to_box_form(default_boxes)
        )

        # which gt box is best matched per default
        max_values = tf.reduce_max(iou_matrix, axis=1)
        max_indices = tf.math.argmax(iou_matrix, axis=1)

        # which default is best matched per gt box
        gt_max_indices = tf.reshape(tf.math.argmax(iou_matrix, axis=0), (-1, 1))

        matches = tf.cast(tf.math.greater_equal(max_values, 0.5), tf.float32)

        max_indices = tf.tensor_scatter_nd_update(max_indices, gt_max_indices, tf.range(gt_max_indices.shape[0], dtype=tf.int64))
        matches = tf.tensor_scatter_nd_update(matches, gt_max_indices, [1] * gt_max_indices.shape[0])

        gt_boxes = tf.cast(tf.gather(ground_truth, max_indices), tf.float32)

        return gt_boxes, matches
    
    @staticmethod
    def calculate_box_offsets(matches, boxes, default_boxes):
        delta_cx = (boxes[:, 0] - default_boxes[:, 0]) / default_boxes[:, 2] * matches
        delta_cy = (boxes[:, 1] - default_boxes[:, 1]) / default_boxes[:, 3] * matches
        delta_w = tf.math.log(boxes[:, 2] / default_boxes[:, 2]) * matches
        delta_h = tf.math.log(boxes[:, 3] / default_boxes[:, 3]) * matches

        delta_m = tf.stack([delta_cx, delta_cy, delta_w, delta_h], axis=-1)

        return delta_m
