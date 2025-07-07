import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

import tqdm


output_classes = {}

# classes = ['ARRANHADURA', 'ARTRITE', 'CARCACA','DERMATITE', 'DERMATOSE', 'FRATURA', 'HEMATOMA', 'RUPTURA']
classes = {'FALHA':0, 'GRAVE':1, 'LEVE':2, 'MODERADA':3, 'MODERADO':3}

def convert(size, box):
    image_width = size[0]
    image_height = size[1]

    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]

    if w < 0:
        w = box[0] - box[1]

    if h < 0:
        h = box[2] - box[3]

    x = x / image_width
    y = y / image_height
    w = w / image_width
    h = h / image_height

    if x < 0:
        x = 0.
    elif x > 1:
        x = 1.

    if y < 0:
        y = 0.
    elif y > 1:
        y = 1.

    if w < 0:
        w = 0.
    elif w > 1:
        w = 1.

    if h < 0:
        h = 0.
    elif h > 1:
        h = 1.

    return (x, y, w, h)


def convert_annotation(input_file, output_file):
    try:
        in_file = open(input_file)
        out_file = open(output_file, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()

        if '20210618-0090-1-0110-0791' in output_file:
            pass

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text

            cls_id = classes[cls]
            # cls_id = classes.index(cls)

            if cls_id != 4:
                if cls_id not  in output_classes:
                    output_classes[cls_id] = 1
                else:
                    output_classes[cls_id] += 1
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        out_file.close()
    except:
        os.remove(input_file)


def generate_labels(dataset_path):

        images = os.path.join(dataset_path, 'IMAGES')
        annotations_path = os.path.join(dataset_path, 'ANNOTATIONS')

        annotations = os.listdir(annotations_path)

        for annotation in tqdm.tqdm(annotations):
            if 'xml' in annotation:
                annotation_xml = os.path.join(annotations_path, annotation)
                # print(annotation_xml)
                annotation_txt = os.path.join(images,
                                              annotation_xml.replace('xml', 'txt').split('/')[-1])
                convert_annotation(annotation_xml, annotation_txt)



if __name__ == '__main__':
    generate_labels('/home/diego/2TB/datasets/eco/new/BRUISE')
    print(output_classes)