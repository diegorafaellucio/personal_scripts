import os
import shutil

import cv2
import tqdm


from base.src.new_classifier.classifier import Classifier
from base.src.new_classifier.classifier_v8 import ClassifierV8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filter_black_list  = ['VIRADA_TOTAL', 'VIRADA_PARCIAL', 'BANDA_A_ANGULADA_COSTELA', 'BANDA_B_ANGULADA_COSTELA']



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



def process_model(images_input_path, images_output_path, classifier_model_path, filter_model_path, device='cuda:0'):
    classifier = ClassifierV8(classifier_model_path)
    filter = Classifier(filter_model_path)


    if not os.path.exists(images_output_path):
        os.makedirs(images_output_path)
    else:
        shutil.rmtree(images_output_path)
        os.makedirs(images_output_path)


    images = os.listdir(images_input_path)


    for image in tqdm.tqdm(images):

        image_path = os.path.join(images_input_path, image)

        img = cv2.imread(image_path)

        filter_results = filter.detect(img)

        filter_best_result = get_best_result(filter_results)

        if filter_best_result is None:

            output_classification_path = os.path.join(images_output_path, '97')

            if not os.path.exists(output_classification_path):
                os.mkdir(output_classification_path)

            new_image_path = os.path.join(output_classification_path, image)
            shutil.copy(image_path, new_image_path)

        elif filter_best_result['label'] not in filter_black_list:


            classification_results = classifier.detect(img)



            best_result = get_best_result(classification_results)

            confidence = 0

            if best_result is not None:
                label = best_result['label']
                confidence = filter_best_result['confidence']
            else:
                label = 'INCLASSIFICAVEL'

            output_classification_path  = os.path.join(images_output_path, label)

            if not os.path.exists(output_classification_path):
                os.mkdir(output_classification_path)

            new_image_path = os.path.join(output_classification_path, '{}_{}'.format(round(confidence, 2), image))

            shutil.copy(image_path, new_image_path)

        else:
            output_classification_path = os.path.join(images_output_path, '97')

            if not os.path.exists(output_classification_path):
                os.mkdir(output_classification_path)

            new_image_path = os.path.join(output_classification_path, image)
            shutil.copy(image_path, new_image_path)


if __name__ == '__main__':
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2023/12/12',
    #               '/home/diego/BARRA_MANSA_12_12_2023_V8_NANO',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/416/arn_2.0_5_classes_v8_nano/weights/best.pt')

    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/01/16/IMAGES',
    #               '/home/diego/VALIDACOES/ARN_1.0_16_01_2024_V8_KEVIN',
    #               '/home/diego/Downloads/best.pt', '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/filter_416_v8_nano_no_augmentation/weights/best.pt')

    process_model('/home/diego/jnb_2',
                  'jnb_2',
                  '/home/diego/Dropbox/Projects/PycharmProjects/Eco/modelos/meat/weight.pt',
                  '/home/diego/Dropbox/Projects/PycharmProjects/Eco/modelos/skeleton/weight.pt')

    # process_model('/home/diego/2TB/datasets/eco/AUDITORIA_BOVINOS/MINERVA/ARN/2024/01/16/IMAGES',
    #               '/home/diego/ARN_arn+mso_2.0_16_01_2024_V8_SMALL_NO_AUGMENTATION',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/arn+mso_2.0_5_classes_v8_small_416_no_augment/weights/best.pt',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/filter/416/filter_416_v8_nano_no_augmentation/weights/best.pt')
    #
    # process_model('/home/diego/2TB/datasets/eco/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/16/',
    #               '/home/diego/ARN_arn+mso_2.0_16_02_2024_V8_SMALL_NO_AUGMENTATION',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/arn+mso_2.0_5_classes_v8_small_416_no_augment/weights/best.pt',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/filter/416/filter_416_v8_nano_no_augmentation/weights/best.pt')



    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/03',
    #               '/home/diego/BARRA_MANSA_03_01_2024_V8_NANO',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/416/arn_2.0_5_classes_v8_nano/weights/best.pt')
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/03',
    #               '/home/diego/BARRA_MANSA_03_01_2024_V8_NANO_NO_AUGMENTATION',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/416/arn_2.0_5_classes_v8_nano_no_augment/weights/best.pt')
