import os
import shutil

import cv2
import tqdm

from ultralytics import YOLO

from base.src.new_classifier.classifier import Classifier
from base.src.new_classifier.classifier_v8 import  ClassifierV8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_best_result(results):
    best_result = None
    best_score = 0

    try:

        for result in results:
            score = result['confidence']
            if score > best_score:
                best_result = result
    except:
        pass

    return best_result



def process_model(test_file_path, images_output_path, classifier_model_path, device='cuda:0'):
    classifier = ClassifierV8(classifier_model_path)


    if not os.path.exists(images_output_path):
        os.mkdir(images_output_path)
    else:
        shutil.rmtree(images_output_path)
        os.mkdir(images_output_path)

    images = None

    with open(test_file_path) as file:
        images = [line.rstrip() for line in file]


    for image_path in tqdm.tqdm(images):

        image_path_elements = image_path.split('/')


        img = cv2.imread(image_path)

        gt_class = image_path_elements[-3]

        gt_output_path = os.path.join(images_output_path, gt_class)

        if not os.path.exists(gt_output_path):
            os.mkdir(gt_output_path)

        classification_results = classifier.detect(img)



        best_result = get_best_result(classification_results)
        confidence = 0

        if best_result is not None:
            label = best_result['label']
            confidence = best_result['confidence']
        else:
            label = 'INCLASSIFICAVEL'

        prediction_path = os.path.join(gt_output_path, label)

        if not os.path.exists(prediction_path):
            os.mkdir(prediction_path)


        new_image_path = os.path.join(prediction_path, '{}_{}'.format(round(confidence, 2), image_path_elements[-1]))

        shutil.copy(image_path, new_image_path)


if __name__ == '__main__':
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2023/12/12',
    #               '/home/diego/BARRA_MANSA_12_12_2023_V8_NANO',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/416/arn_2.0_5_classes_v8_nano/weights/best.pt')


    process_model('/home/diego/2TB/datasets/eco/BOVINOS/TREINOS/DATA_CONFORMACAO_BELEN_1.0_4_CLASSES/test.txt',
                  '/home/diego/2TB/resultado_conformacao_4_classes',
                  '/home/diego/2TB/yolo/Trains/v8/conformation/BELEN_1.0/trains/nano/416/runs/detect/train_augmented/weights/best.pt')



    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/03',
    #               '/home/diego/BARRA_MANSA_03_01_2024_V8_NANO',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/416/arn_2.0_5_classes_v8_nano/weights/best.pt')
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/03',
    #               '/home/diego/BARRA_MANSA_03_01_2024_V8_NANO_NO_AUGMENTATION',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/416/arn_2.0_5_classes_v8_nano_no_augment/weights/best.pt')
