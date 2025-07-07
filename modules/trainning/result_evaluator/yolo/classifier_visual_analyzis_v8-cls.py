import os
import shutil

import cv2
import tqdm

from ultralytics import YOLO

from base.src.new_classifier.classifier import Classifier
from base.src.new_classifier.classifier_v8 import ClassifierV8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

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





def process_model(images_input_path, images_output_path, classifier_model_path, device='cuda:0'):
    classifier = ClassifierV8(classifier_model_path)


    if not os.path.exists(images_output_path):
        os.mkdir(images_output_path)
    else:
        shutil.rmtree(images_output_path)
        os.mkdir(images_output_path)


    images = os.listdir(images_input_path)


    for image in tqdm.tqdm(images):

        image_path = os.path.join(images_input_path, image)

        img = cv2.imread(image_path)

        img_pil = Image.fromarray(img)

        classification_results = classifier(img_pil)[0]

        probs = classification_results.probs.cpu().data.numpy()

        class_id = np.argmax(probs)

        class_names = classification_results.names

        predicted_class = class_names[class_id]

        output_classification_path  = os.path.join(images_output_path, predicted_class)

        if not os.path.exists(output_classification_path):
            os.mkdir(output_classification_path)

        new_image_path = os.path.join(output_classification_path, image)

        shutil.copy(image_path, new_image_path)


if __name__ == '__main__':
    process_model('//home/diego/2TB/datasets/eco/AUDITORIA_BOVINOS/BARRA_MANSA/2024/03/13',
                  'v8',
                  '/home/diego/2TB/yolo/Trains/train_scripts/v8/meat/416/ARN1.0+MSO+SUL_BEEF+RIO_MARIA/runs/detect/arn1.0+mso+sul_beef+rio_maria_meat_nano_v5_416_no_augmentation/weights/best.pt')


    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/01',
    #               '/home/diego/CLS_ARN_01_02_2024_V8_NANO_NO_AUGMENTATION',
    #               '/home/diego/2TB/yolo/Trains/train_scripts/runs/classify/arn_1.0_5_classes_v8_nano_no_augment-cls/weights/best.pt')
