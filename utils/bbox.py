import numpy as np
import os
import cv2
from .colors import get_color

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=2)
            #cv2.fillPoly(img=image, pts=[region], color=get_color(label)) # do not fill polygon because it covers too large area
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=5e-4 * image.shape[0], 
                        color=get_color(label), #(0,0,0), 
                        thickness=2)
        
    return image  

def draw_anno_boxes(image, annotations):
    for anno in annotations:
#        region = np.array([[anno[0]-3,        anno[1]], 
#                           [anno[0]-3,        anno[1]-anno[3]+anno[1]-26], 
#                           [anno[2]+13, anno[1]-anno[3]+anno[1]-26], 
#                           [anno[2]+13, anno[1]]], dtype='int32')  
        
        cv2.rectangle(img=image, pt1=(anno[0],anno[1]), pt2=(anno[2],anno[3]), color=(0,128,0), thickness=2)
        #cv2.fillPoly(img=image, pts=[region], color=get_color(label))
#        cv2.putText(img=image, 
#                    text=str(anno[4]), 
#                    org=(anno[0]+13, anno[1] - 13), 
#                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#                    fontScale=1e-3 * image.shape[0], 
#                    color=(0,128,0), 
#                    thickness=1)
    return image  

def write_predict_boxes_xml(boxes, output_path, image_path, image, labels, obj_thresh):
    image_path = image_path.split('/')[-1]
    from lxml import etree as ET
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = image_path
    ET.SubElement(annotation, "path").text = 'Unkown'
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = 'Unknown'
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image.shape[0])
    ET.SubElement(size, "height").text = str(image.shape[1])
    ET.SubElement(size, "depth").text = str(image.shape[2])
    ET.SubElement(annotation, "segmented").text =str(0)
    count = 0
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
                
        if label >= 0:
            myobject = ET.SubElement(annotation, "object",name="detection"+str(count))
            ET.SubElement(myobject, "name").text = 'fish'
            ET.SubElement(myobject, "pose").text = 'unspecified'
            ET.SubElement(myobject, "truncated").text = str(1)
            ET.SubElement(myobject, "difficult").text = str(0)
            bndbox = ET.SubElement(myobject, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(box.xmin)
            ET.SubElement(bndbox, "ymin").text = str(box.ymin)
            ET.SubElement(bndbox, "xmax").text = str(box.xmax)
            ET.SubElement(bndbox, "ymax").text = str(box.ymax)
            ET.SubElement(bndbox, "confidence").text = str(box.classes[0])
            count = count + 1
    tree = ET.ElementTree(annotation)
    tree.write(output_path + image_path[0:-3]+".xml",  pretty_print=True)
    return None