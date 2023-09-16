import numpy as np
import pandas as pd
import cv2
import random
import os
import xml.etree.ElementTree as ET
import math

class InkMLParser():
    '''
    Parses InkML data into images, labels, and bounding boxes
    '''
    label_codes = ["0","1","2","3","4","5","6","7","8","9","+","-","\\times","\\div","=","(",")"]

    def __init__(self):
        pass
    
    def get_traces_data(self, inkml_file_abs_path):
        '''
        Trace processor adapted from from https://www.kaggle.com/code/kalikichandu/preprossing-inkml-to-png-files/notebook
        '''

        traces_data = []
        
        # tries to parse ink ml file to extract XML tree
        try:
            tree = ET.parse(inkml_file_abs_path)
            root = tree.getroot()
            doc_namespace = "{http://www.w3.org/2003/InkML}"
        except:
            return None

        # navigates tree to find all traces in file
        traces_all = [
            {
                'id': trace_tag.get('id'),
                'coords': [
                    (
                        np.array(
                            [float(axis_coord) * 25 for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') else
                            [float(axis_coord) * 25 for axis_coord in coord.split(' ')]
                        )
                    )
                    for coord in (trace_tag.text).replace('\n', '').split(',')
                ]
            }
            for trace_tag in root.findall(doc_namespace + 'trace')
        ]

        # 'Sort traces_all list by id to make searching for references faster'
        traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

        # 'Always 1st traceGroup is a redundant wrapper'
        traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

        if traceGroupWrapper is not None:
            for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

                label = traceGroup.find(doc_namespace + 'annotation').text

                # 'traces of the current traceGroup'
                traces_curr = []
                for traceView in traceGroup.findall(doc_namespace + 'traceView'):

                    # 'Id reference to specific trace tag corresponding to currently considered label'
                    traceDataRef = int(traceView.get('traceDataRef'))

                    # 'Each trace is represented by a list of coordinates to connect'
                    if traceDataRef < len(traces_all):
                        single_trace = traces_all[traceDataRef]['coords']
                        traces_curr.append(single_trace)

                # check traces are valid
                if len(traces_curr):
                    traces_data.append({'label': label, 'trace_group': traces_curr})

        else:
            # 'Consider Validation data that has no labels'
            [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

        return np.array(traces_data)
    
    def inkml_to_labelled_image(self, input_path):
        '''
        Parses InkML file specified by input path into an image
        '''

        # et traces from XML
        equation = self.get_traces_data(input_path)

        # checks if parsed data is valid
        if not isinstance(equation, np.ndarray):
            return None

        points = np.array([points for symbol in equation for stroke in symbol["trace_group"] for points in stroke])

        if points.shape[0] == 0:
            return None
        
        max_x, min_x =  np.max(points[:,1]), np.min(points[:,1])
        max_y, min_y =  np.max(points[:,0]), np.min(points[:,0])

        points[:,1] -= min_x
        points[:,0] -= min_y

        max_dim = max(max_x - min_x, max_y - min_y)

        MAX_INNER_SIZE = 268
        PADDING = 16
        MAX_SIZE = MAX_INNER_SIZE + PADDING * 2

        points = (points / max_dim * MAX_INNER_SIZE).astype(int)
        
        img = np.zeros((np.max(points[:,1]) + PADDING * 2, np.max(points[:,0]) + PADDING * 2), np.float32)

        bounding_boxes = []
        labels = []

        x_left, x_right = math.floor((MAX_SIZE - img.shape[0]) / 2), math.ceil((MAX_SIZE - img.shape[0]) / 2)
        y_top, y_bottom = math.floor((MAX_SIZE - img.shape[1]) / 2), math.ceil((MAX_SIZE - img.shape[1]) / 2)

        # calculates bounding box and draws
        for symbol in equation:

            x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0

            for stroke in symbol["trace_group"]:
                stroke = np.array(stroke)

                # temp coordinates
                x3, y3 = np.min(stroke[:,1]) - min_x, np.min(stroke[:,0]) - min_y
                x4, y4 = np.max(stroke[:,1]) - min_x, np.max(stroke[:,0]) - min_y

                (x3, y3, x4, y4) =  np.array((x3, y3, x4, y4) / max_dim * min(MAX_INNER_SIZE, max_x, max_y) + PADDING).astype(int)

                x1, y1 = min(x1, x3), min(y1, y3)
                x2, y2 = max(x2, x4), max(y2, y4)

                for point1, point2 in zip(stroke, stroke[1:]):
                    point1 = (np.array((point1[0] - min_y, point1[1] - min_x)) / max_dim * min(MAX_INNER_SIZE, max_x, max_y) + PADDING).astype(int)
                    point2 = (np.array((point2[0] - min_y, point2[1] - min_x)) / max_dim * min(MAX_INNER_SIZE, max_x, max_y) + PADDING).astype(int)
                    img = cv2.line(img, point1, point2, 1)
            
            bounding_boxes.append(((x1 + x2) / 2 + x_left, (y1 + y2) / 2 + y_top, x2 - x1 + 25, y2 - y1 + 25))
            labels.append(symbol["label"])

        img = np.pad(img, ((x_left, x_right), (y_top, y_bottom)), mode="constant", constant_values=0)

        return [img, labels, bounding_boxes]
    
    def reorganize_labels(self, labels):
        coded_labels = []

        for label in labels:
            if not label in self.label_codes:
                self.label_codes.append(label)
            
            coded_labels.append(self.label_codes.index(label))

        return self.label_codes, coded_labels