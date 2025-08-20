import os
import shutil

import cv2
import tqdm

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from base.src.ml.classifier.ultralytics_classifier.classifier import Classifier
from base.src.utils.utils import generate_json
import json
device = 'cuda:0'

padding = 550

classifier = Classifier('/home/diego/Projects/ECO/ecoia-classifier/data/models/skeleton/weight.pt')
print()
# side_classifier = Classifier('/home/diego/1TB/ecotrace/frameworks/yolov5/runs/train/exp4/weights/last.pt', device=device, max_det=10)


def process_dataset(dataset_path):
    subsets = os.listdir(dataset_path)

    for subset in tqdm.tqdm(subsets):
        subset_path = os.path.join(dataset_path, subset)
        generate_jsons(subset_path, subset.upper())

def generate_jsons(dataset_path, label):
    executor = ThreadPoolExecutor(max_workers=50)

    futures = []

    dataset_path = dataset_path


    images_path = os.path.join(dataset_path, 'IMAGES')
    annotations_path = os.path.join(dataset_path, 'ANNOTATIONS_JSON')
    not_annotated_path = os.path.join(dataset_path, 'NOT_ANNOTATED_IMAGES')

    if os.path.exists(annotations_path):
        shutil.rmtree(annotations_path)
        os.mkdir(annotations_path)
    else:
        os.mkdir(annotations_path)


    if not os.path.exists(not_annotated_path):
        os.mkdir(not_annotated_path)

    images = os.listdir(images_path)

    image_counter = 0

    for image_name in tqdm.tqdm(images):

        image_counter += 1

        image_path = os.path.join(images_path, image_name)

        try:

            image_name = os.path.splitext(image_name)[0]

            json_file_name = '{}.json'.format(image_name)

            json_path = os.path.join(annotations_path, json_file_name)




            counter = '{}/{}'.format(image_counter, len(images))

            # output_json = generate_json(label, [x_min, y_min], [x_max, y_max], image, img_width, img_height)
            futures.append(executor.submit(generate_json, label, image_name, json_path, counter, padding, image_path, classifier, not_annotated_path))
            # with open(json_path, "w") as outfile:
            #     json.dump(output_json, outfile, indent=4)

            performed_counter = 0


        except Exception as ex:
            pass
            # os.remove(image_path)

    for future in concurrent.futures.as_completed(futures):
        performed_counter += 1

        print('{}/{}'.format(performed_counter, len(images)))




if __name__ == '__main__':

    process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/5.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/7.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PRN/4.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/SUL_BEEF/SB/6.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/7.0')

    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BLN/3.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/BARRA_MANSA/BM/4.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/JBO/1.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/JNB/1.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PGO/3.0')
    # process_dataset('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/RIO_MARIA/RM/1.0')



