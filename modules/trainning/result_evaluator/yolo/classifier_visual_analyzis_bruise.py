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




def process_model(images_input_path, images_output_path, classifier_model_path, device='cuda:0'):
    classifier =  classifier = Classifier(classifier_model_path,
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

        classification_results = classifier.detect(img)

        best_result = get_best_result(classification_results)

        if best_result is not None:
            label = best_result['label']
        else:
            label = 'INCLASSIFICAVEL'

        output_classification_path  = os.path.join(images_output_path, label)

        if not os.path.exists(output_classification_path):
            os.mkdir(output_classification_path)

        new_image_path = os.path.join(output_classification_path, image)

        shutil.copy(image_path, new_image_path)


if __name__ == '__main__':
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/validacao_arn', '/home/diego/validacao_arn_1_classe', '/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_novo_arn_1_classe/weights/best.pt')
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/2-FILTER/ECOTRACE/AUDITAR/BTS', '/home/diego/BTS_PROCESSADO_OLD', '/home/diego/2TB/yolo/new_v5/runs/train/eco_filter_v2/weights/best.pt')
    process_model('/home/diego/2TB/validacao_treino_eco/BM',
                  '/home/diego/2TB/validacao_treino_eco/resultados/443',
                  '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3_BM9/trains/nano/640/runs/detect/train_nano_1.0_sgd/weights/best.pt')
