import os
import shutil

import torch
from PIL import Image
import cv2
import tqdm

from ultralytics import YOLO

from base.src.new_classifier.classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from base.src.classifier.utils.augmentations import letterbox


filter_black_list  = ['VIRADA_TOTAL', 'VIRADA_PARCIAL', 'BANDA_A_ANGULADA_COSTELA', 'BANDA_B_ANGULADA_COSTELA']
first_stage_class_dict  = {'1':'1', '2':'3', '3':'5',}
second_stage_class_dict  = {'1':'2', '2':'4'}
def get_best_result(results):
    boxes = results[0].boxes.xyxy.tolist()
    class_ids = results[0].boxes.cls.tolist()
    class_names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    best_result = None
    best_score = 0

    for index, confidence in enumerate(confidences):
        class_id = int(class_ids[index])
        score = confidence
        if score > best_score:
            best_result = class_names[class_id]

    return best_result




def process_model(images_input_path, images_output_path, first_stage_classifier_model_path, second_stage_classifier_model_path, filter_model_path, device='cuda:0'):
    first_stage_classifier = YOLO(first_stage_classifier_model_path)
    second_stage_classifier = YOLO(second_stage_classifier_model_path)
    filter = YOLO(filter_model_path)


    if not os.path.exists(images_output_path):
        os.mkdir(images_output_path)
    else:
        shutil.rmtree(images_output_path)
        os.mkdir(images_output_path)


    images = os.listdir(images_input_path)



    for image in tqdm.tqdm(images):


        image_path = os.path.join(images_input_path, image)

        img = cv2.imread(image_path)

        filter_results = filter.predict(img)

        filter_best_result = get_best_result(filter_results)

        if filter_best_result not in filter_black_list:


            classification_results = first_stage_classifier.predict(img)[0]

            probs = classification_results.probs.cpu().data.numpy()

            class_id = np.argmax(probs)

            class_names = classification_results.names

            predicted_class = class_names[class_id]


            if predicted_class == '4':
                second_stage_classification_results = second_stage_classifier.predict(img)[0]

                second_stage_probs = second_stage_classification_results.probs.cpu().data.numpy()

                second_stage_class_id = np.argmax(second_stage_probs)

                second_stage_class_names = second_stage_classification_results.names

                second_stage_predicted_class = second_stage_class_names[second_stage_class_id]
                second_stage_predicted_class = second_stage_class_dict[second_stage_predicted_class]

                output_classification_path = os.path.join(images_output_path, second_stage_predicted_class)

                if not os.path.exists(output_classification_path):
                    os.mkdir(output_classification_path)

                new_image_path = os.path.join(output_classification_path, image)

                shutil.copy(image_path, new_image_path)

            else:
                predicted_class = first_stage_class_dict[predicted_class]
                output_classification_path  = os.path.join(images_output_path, predicted_class)

                if not os.path.exists(output_classification_path):
                    os.makedirs(output_classification_path)

                new_image_path = os.path.join(output_classification_path, image)

                shutil.copy(image_path, new_image_path)


if __name__ == '__main__':
    process_model('/home/diego/2TB/datasets/eco/AUDITORIA_BOVINOS/MINERVA/MSO/2024/02/13/NO_BACKGROUND_IMAGES',
                  '/home/diego/CLS_MSO_13_02_2024_V8_NANO_WITH_FILTER',
                  # '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment_cls_1_step_v2/weights/best.pt',
                  # '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment-cls/weights/best.pt',
                  '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment_cls_1_step_v2/weights/best.pt',
                  '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment_cls_2_step_v2/weights/best.pt',
                  '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/filter/416/filter_416_v8_nano_no_augmentation/weights/best.pt'
                  )


    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/MSN/2024/02/13/IMAGES',
    #               '/home/diego/CLS_ARN_16_02_2024_V8_NANO_WITH_FILTER',
    #               # '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment_cls_1_step_v2/weights/best.pt',
    #               # '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment-cls/weights/best.pt',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/v8/meat/416/runs/classify/arn_1.0_5_classes_v8_nano_cls_1_step_v2/weights/best.pt',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/v8/meat/416/runs/classify/arn_1.0_5_classes_v8_nano_cls_2_step_v2/weights/best.pt',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/detect/filter/416/filter_416_v8_nano_no_augmentation/weights/best.pt'
    #               )




    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/01',
    #               '/home/diego/CLS_ARN_01_02_2024_V8_NANO_NO_AUGMENTATION',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment-cls/weights/best.pt')
