import json
import traceback

import numpy as np
import cv2
import os
import shutil
import xml.etree.ElementTree as ET

import tqdm

label_dict = {}


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def generate_xml(annotation_file_path, output_dir):
    file_name = annotation_file_path.split('/')[-1]

    if 'xml' in annotation_file_path:
        file_output_path = os.path.join(output_dir, file_name)
        shutil.copy(annotation_file_path, file_output_path)

    else:

        file_name = file_name.replace('json', 'xml')

        file_output_path = os.path.join(output_dir, file_name)

        with open(annotation_file_path) as json_file:

            try:

                data = json.load(json_file)

                file_path = data['imagePath']
                image_file_name = file_path.split('/')[-1]

                width_size = data['imageWidth']
                height_size = data['imageHeight']

                xmin_value = int(data['shapes'][0]['points'][0][0])
                ymin_value = int(data['shapes'][0]['points'][0][1])
                xmax_value = int(data['shapes'][0]['points'][1][0])
                ymax_value = int(data['shapes'][0]['points'][1][1])

                object_name = data['shapes'][0]['label']

                # print(path_string)

                label_dict[object_name] = 'OK'

                root = ET.Element('annotation')

                folder = ET.SubElement(root, "folder")
                folder.text = 'IMAGES'

                filename = ET.SubElement(root, "filename")
                filename.text = image_file_name

                path = ET.SubElement(root, "path")
                path.text = file_path

                source = ET.SubElement(root, "source")
                database = ET.SubElement(source, "database")
                database.text = 'Unknown'

                size = ET.SubElement(root, "size")
                width = ET.SubElement(size, "width")
                width.text = str(width_size)
                height = ET.SubElement(size, "height")
                height.text = str(height_size)
                depth = ET.SubElement(size, "depth")
                depth.text = str(3)

                segmented = ET.SubElement(root, "segmented")
                segmented.text = '0'

                object = ET.SubElement(root, 'object')

                name = ET.SubElement(object, "name")
                name.text = object_name

                pose = ET.SubElement(object, "pose")
                pose.text = "Unspecified"

                truncated = ET.SubElement(object, "truncated")
                truncated.text = "0"

                difficult = ET.SubElement(object, "difficult")
                difficult.text = "0"

                bndbox = ET.SubElement(object, "bndbox")

                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(xmin_value)

                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(ymin_value)

                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(xmax_value)

                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(ymax_value)

                indent(root)

                # ET.dump(root)

                # print(file_output_path)
                ET.ElementTree(root).write(file_output_path)
            except Exception as ex:
                file_name = file_name.replace('json', 'xml')
                file_output_path = os.path.join(output_dir, file_name)
                shutil.copy(annotation_file_path, file_output_path)


def process_directory(directory, output_dir):
    files = [annotation for annotation in os.listdir(directory)]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file in tqdm.tqdm(files):
        # print('{}/{}'.format(i+1, len(files)))
        generate_xml(os.path.join(directory, file), output_dir)


if __name__ == '__main__':
    process_directory(
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/AUSENTE/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/AUSENTE/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/ESCASSA/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/ESCASSA/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/MEDIANA/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/MEDIANA/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/UNIFORME/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/UNIFORME/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/EXCESSIVA/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO/general/EXCESSIVA/ANNOTATIONS')
