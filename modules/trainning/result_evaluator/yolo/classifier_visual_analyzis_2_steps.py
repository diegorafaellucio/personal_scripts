import os
import shutil

import cv2
import tqdm

from base.src.new_classifier.classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_best_result(results):
    best_result = None
    best_score = 0

    for result in results:
        score = result['confidence']
        if score > best_score:
            best_result = result

    return best_result




def process_model(images_input_path, images_output_path, step_1_classifier_model_path, step_2_classifier_model_path, device='cuda:0', step_1_white_list= ['AUSENTE', 'ESCASSA', 'MEDIANA', 'MEDIANDA'], step_2_white_list= ['UNIFORME', 'EXCESSIVA']):

    step_1_classifier = Classifier(step_1_classifier_model_path,
                                   device=device, max_det=10)

    step_2_classifier = Classifier(step_2_classifier_model_path,
                                  device=device, max_det=10)


    if not os.path.exists(images_output_path):
        os.mkdir(images_output_path)
    else:
        shutil.rmtree(images_output_path)
        os.mkdir(images_output_path)


    images = os.listdir(images_input_path)


    for image in tqdm.tqdm(images):

        image_path = os.path.join(images_input_path, image)

        img = cv2.imread(image_path)

        step_1_classification_results = step_1_classifier.detect(img)

        step_1_best_result = get_best_result(step_1_classification_results)

        confidence = 0.0

        if step_1_best_result is not None:
            label = step_1_best_result['label']
            confidence = step_1_best_result['confidence']

            if label not in step_1_white_list:
                step_2_classification_results = step_2_classifier.detect(img)

                step_2_best_result = get_best_result(step_2_classification_results)



                if step_2_best_result is not None:
                    label = step_2_best_result['label']
                    confidence = step_2_best_result['confidence']

                else:
                    label = 'INCLASSIFICAVEL'
                    confidence = 0.0


        else:
            label = 'INCLASSIFICAVEL'



        output_classification_path  = os.path.join(images_output_path, label)

        if not os.path.exists(output_classification_path):
            os.mkdir(output_classification_path)

        new_image_name = '{}_{}'.format(confidence, image)

        new_image_path = os.path.join(output_classification_path, new_image_name)

        shutil.copy(image_path, new_image_path)


if __name__ == '__main__':
    process_model('/home/diego/Downloads/MEDIANA_ARN_HOJE', 'MEDIANA_ARN_HOJE', '/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_general_step12/weights/best.pt', '/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_general_step28/weights/best.pt')
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/01/16/IMAGES', '/home/diego/ARN_16_01_2024_CLASSIFICATION_2_STEPS', '/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_general_step12/weights/best.pt', '/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_general_step28/weights/best.pt')
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MSO/2024/01/13', '/home/diego/MSO_2024_01_13', '/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_general_step12/weights/best.pt', '/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_general_step28/weights/best.pt')
