import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

import tqdm


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


def convert_annotation(input_file, output_file, classes):
    try:
        in_file = open(input_file)
        out_file = open(output_file, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            
            # Try to find the class name using both 'name' and 'n' tags
            name_tag = obj.find('name')
            if name_tag is None:
                name_tag = obj.find('n')
                if name_tag is None:
                    # Silently skip objects without name tags
                    continue
                    
            cls = name_tag.text
            cls = cls.upper().replace(" ", "_")

            # Check if the class is in our classes dictionary
            if cls not in classes:
                # Silently skip classes not in our dictionary
                continue
                
            cls_id = classes[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        out_file.close()
        in_file.close()
    except Exception as ex:
        print(f"Error processing {input_file}: {str(ex)}")
        pass


def generate_labels(dataset_path, classes, image_path='IMAGES'):
    images = os.path.join(dataset_path, image_path)
    annotations_path = os.path.join(dataset_path, 'ANNOTATIONS')

    annotations = os.listdir(annotations_path)

    for annotation in tqdm.tqdm(annotations):
        if 'xml' in annotation:
            annotation_xml = os.path.join(annotations_path, annotation)
            # print(annotation_xml)
            annotation_txt = os.path.join(images,
                                          annotation_xml.replace('xml', 'txt').split('/')[-1])
            convert_annotation(annotation_xml, annotation_txt, classes)


if __name__ == '__main__':

    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/RLM/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BLN/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0',
    # #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/MSO/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/FRIGOL/LP/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/ARN/2.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/RIO_MARIA/RM/2.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PRN/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BTS/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})

    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/RLM/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BLN/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0',
    # #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/MSO/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/FRIGOL/LP/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/ARN/2.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/RIO_MARIA/RM/2.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PRN/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BTS/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/SULBEEF/SB/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})

    generate_labels('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/6.0',
                    classes={"MEAT": 0, "PERSON": 1, "SCRAP": 2, "HELMET": 3, "TRASH": 4, 'FULL_TRAY': 5, 'EMPTY_TRAY': 6})
    #
    # generate_labels('/home/diego/2TB/datasets/COGTIVE/MAURICEA_MONITORAMENTO/1.0',
    #                 classes={"FILE_DE_FRANGO": 0, "PERSON": 1, "SASSAMI": 2, "BANDEIJA": 3, "FACA": 4, 'RETALHOS_SACO': 5,
    #                          'SACO_OSSO': 6})
    #
    # generate_labels('/home/diego/2TB/datasets/COGTIVE/BIG_CHARQUE/1.0',
    #                 classes={"PERSON": 0, "CHARQUE_1KG": 1, "CHARQUE_500G": 2, "RETALHOS": 3, "CHARQUES_PULMAO": 4})
    #
    # generate_labels('/home/diego/2TB/datasets/COGTIVE/CANAA_AMBIENTAL/1.0',
    #                 classes={"PERSON": 0})
    #
    # generate_labels('/home/diego/2TB/datasets/COGTIVE/CANAA_NORTE/1.0',
    #                 classes={"PERSON": 0})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})
    #
    # generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0',
    #                 classes={"FALHA": 0, "LEVE": 1, "MODERADA": 2, "GRAVE": 3, "GRAVE_ABCESSO": 4, 'GRAVE_ABSCESSO': 4})



# generate_labels('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/3.0',
    #                 image_path='IMAGES', classes={"MEAT": 0, "PERSON": 1})
    #
    #
    # generate_labels('/home/diego/2TB/datasets/COGTIVE/MAURICEA/MONITORAMENTO/1.0',
    #                 classes={"FILE_DE_FRANGO": 0,"PERSON": 1,"SASSAMI": 2,"BANDEIJA": 3,"FACA": 4,"RETALHOS_SACO": 5,"SACO_OSSO": 6})



