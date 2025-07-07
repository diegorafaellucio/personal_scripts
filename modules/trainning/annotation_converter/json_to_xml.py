import json
import numpy as np
import cv2
import os
import shutil
import xml.etree.ElementTree as ET

label_dict = {}


# allowed_shapes = ['artrite']

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


def generate_xml(json_file_path, output_dir):
    with open(json_file_path) as json_file:

        try:
            data = json.load(json_file)
            json_file_name = json_file_path.split('/')[-1]

            file_path = data['imagePath']
            image_file_name = file_path.split('/')[-1]

            width_size = data['imageWidth']
            height_size = data['imageHeight']

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

            object_counter = 0

            if len(data['shapes']) == 0:
                return

            for shape in data['shapes']:
                object = ET.SubElement(root, 'object')

                xmin_value = int(shape['points'][0][0])
                ymin_value = int(shape['points'][0][1])
                xmax_value = int(shape['points'][1][0])
                ymax_value = int(shape['points'][1][1])

                label = str(shape['label']).upper().replace(']', '')

                # print(path_string)
                if label not in label_dict:

                    label_dict[label] = 1
                else:
                    label_dict[label] += 1

                name = ET.SubElement(object, "name")
                name.text = str(label).upper()

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

            # if object_counter > 0:

            indent(root)

            # ET.dump(root)
            file_name = json_file_name.replace('json', 'xml')

            file_output_path = os.path.join(output_dir, file_name)
            # print(file_output_path)
            ET.ElementTree(root).write(file_output_path)
        except Exception as ex:
            print(ex)
            pass


def process_directory(directory, output_dir):
    files = [annotation for annotation in os.listdir(directory) if 'json' in annotation]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    else:
        os.mkdir(output_dir)

    for i, file in enumerate(files):
        print('{}/{}'.format(i + 1, len(files)))
        generate_xml(os.path.join(directory, file), output_dir)


if __name__ == '__main__':
    # process_directory(
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/AUSENTE/ANNOTATIONS_JSON',
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/AUSENTE/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/ESCASSA/ANNOTATIONS_JSON',
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/ESCASSA/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/MEDIANA/ANNOTATIONS_JSON',
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/MEDIANA/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/UNIFORME/ANNOTATIONS_JSON',
    #     '/home/diego/Downloads/Acabamento_novo_BM_3/UNIFORME/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/ESCASSA/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/ESCASSA/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/MEDIANA/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/MEDIANA/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/UNIFORME/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/UNIFORME/ANNOTATIONS')

    process_directory(
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/EXCESSIVA/ANNOTATIONS_JSON',
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0/EXCESSIVA/ANNOTATIONS')




    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/7.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/7.0/ANNOTATIONS')

    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BLN/1.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BLN/1.0/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/FRIGOL/LP/1.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/FRIGOL/LP/1.0/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/MSO/1.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/MSO/1.0/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/RIO_MARIA/RM/3.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/RIO_MARIA/RM/3.0/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/RLM/1.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/RLM/1.0/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PRN/1.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PRN/1.0/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BTS/1.0/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BTS/1.0/ANNOTATIONS')





    # process_directory(
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/2-FILTER/MINERVA/PGO/1.0/BANDA_B_NORMAL/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/2-FILTER/MINERVA/PGO/1.0/BANDA_B_NORMAL/ANNOTATIONS')


    # process_directory(
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/ESCASSA/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/ESCASSA/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/MEDIANA/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/MEDIANA/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/UNIFORME/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/UNIFORME/ANNOTATIONS')
    #
    # process_directory(
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/EXCESSIVA/ANNOTATIONS_JSON',
    #     '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/EXCESSIVA/ANNOTATIONS')







