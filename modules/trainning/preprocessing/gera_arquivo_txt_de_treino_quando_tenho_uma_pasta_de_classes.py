import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

import tqdm
import cv2


def convert(size, box):
    image_width = size[0]
    image_height = size[1]

    # Correct Pascal VOC to YOLO conversion
    # box = [xmin, ymin, xmax, ymax] (Pascal VOC format)
    # YOLO format needs: [x_center, y_center, width, height] (normalized)
    
    x_center = (box[0] + box[2]) / 2.0  # (xmin + xmax) / 2
    y_center = (box[1] + box[3]) / 2.0  # (ymin + ymax) / 2
    width = box[2] - box[0]             # xmax - xmin
    height = box[3] - box[1]            # ymax - ymin

    # Normalize to image dimensions
    x = x_center / image_width
    y = y_center / image_height
    w = width / image_width
    h = height / image_height

    # Clamp values to [0, 1] range
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


def find_image(images_dir, stem, exts=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
    for ext in exts:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def convert_annotation(input_file, output_file, classes, images_dir):
    try:
        in_file = open(input_file)
        out_file = open(output_file, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()

        # Get intended image size from XML (fallback)
        xml_size = root.find('size')
        xml_w = int(xml_size.find('width').text) if xml_size is not None else None
        xml_h = int(xml_size.find('height').text) if xml_size is not None else None

        # Determine image path to read actual size
        filename_tag = root.find('filename')
        if filename_tag is not None and filename_tag.text:
            image_filename = filename_tag.text
            stem = os.path.splitext(os.path.basename(image_filename))[0]
        else:
            stem = os.path.splitext(os.path.basename(input_file))[0]
        img_path = find_image(images_dir, stem)

        img_w = xml_w
        img_h = xml_h
        if img_path is not None:
            img = cv2.imread(img_path)
            if img is not None:
                img_h, img_w = img.shape[0], img.shape[1]
        # Fallback if we couldn't determine actual image size
        if img_w is None or img_h is None:
            # If XML didn't have size and we couldn't read the image, skip
            in_file.close()
            out_file.close()
            return

        for obj in root.iter('object'):
            diff_tag = obj.find('difficult')
            difficult = diff_tag.text if diff_tag is not None else '0'

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
            # Order must be (xmin, ymin, xmax, ymax) for convert()
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymax').text),
            )
            # Normalize using actual image dimensions
            bb = convert((img_w, img_h), b)
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
            convert_annotation(annotation_xml, annotation_txt, classes, images)


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

    # generate_labels('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/6.0',
    #                 classes={"MEAT": 0, "PERSON": 1, "SCRAP": 2, "HELMET": 3, "TRASH": 4, 'FULL_TRAY': 5, 'EMPTY_TRAY': 6})
    #
    # generate_labels('/home/diego/2TB/datasets/COGTIVE/MAURICEA_MONITORAMENTO/1.0',
    #                 classes={"FILE_DE_FRANGO": 0, "PERSON": 1, "SASSAMI": 2, "BANDEIJA": 3, "FACA": 4, 'RETALHOS_SACO': 5,
    #                          'SACO_OSSO': 6})
    #
    generate_labels('/home/diego/2TB/datasets/COGTIVE/BIG_CHARQUE/2.0',
                    classes={"PERSON": 0, "CHARQUE_1KG": 1, "CHARQUE_500G": 1, "CHARQUE_400G": 1, "RETALHOS": 2, "CHARQUES_PULMAO": 1, "CHARQUE": 1})
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

    # generate_labels('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/9.0',
    #                 image_path='IMAGES', classes={"MEAT": 0, "PERSON": 1, "SCRAP": 2, "HELMET": 3, "TRASH": 4, "FULL_TRAY": 5, "EMPTY_TRAY": 6})

    # generate_labels('/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0',
    #                 image_path='IMAGES',
    #                 classes={"BRAZILIAN_PLATE": 0, "MERCOSUR_PLATE": 1})

    # generate_labels('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/7.0',
    #                 image_path='IMAGES', classes={"MEAT": 0, "PERSON": 1})
    #
    # #
    # generate_labels('/home/diego/2TB/datasets/COGTIVE/MAURICEA_MONITORAMENTO/2.0',
    #                 classes={"FILE_DE_FRANGO": 0,"PERSON": 1,"SASSAMI": 2,"BANDEIJA": 3,"FACA": 4,"RETALHOS_SACO": 5,"SACO_OSSO": 6})

    # generate_labels('/home/diego/2TB/datasets/COGTIVE/MAURICEA_MONITORAMENTO/2.0',
    #                 classes={"FILE_DE_FRANGO": 0,"PERSON": 1,"SASSAMI": 2,"FACA": 3,"RETALHOS_SACO": 4,"SACO_OSSO": 5})

    # generate_labels('/home/diego/2TB/datasets/COGTIVE/Pancristal/1.0',
    #                 classes={"BAGUETE": 0,
    #                          "BAGUETE_MAIOR": 0,
    #                          "BISNAGUINHA": 0,
    #                          "BISNAGUINHA_MENOR": 0,
    #                          "PAO_QUADRADO": 0,
    #                          "PAO_DE_FORMA": 0,
    #                          "PAO_FRANCES": 0,
    #                          "PAO_FRANCES_GD": 0,
    #                          "PAO_FRANCES_PQ": 0,
    #                          "PAO_GD_ESPECIFIC": 0,
    #                          "PAO_HOTDOG": 0,
    #                          "PAO_RETANGULAR": 0,
    #                          "PAO_RETANGULAR_GD": 0,
    #                          "PAO_QUADRADO_GD": 0,
    #                          "PAO_DOCE_450": 0,
    #                          })
