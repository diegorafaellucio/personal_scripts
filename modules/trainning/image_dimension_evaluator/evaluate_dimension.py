import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import tqdm

dataset_path = '/home/diego/2TB/datasets/eco/BOVINOS/GT/MSO/general'

subsets = os.listdir(dataset_path)
#, mso

def get_dimension(image_path, image_counter):
    try:
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        return height, width, image_path, image_counter
    except:
        os.remove(image_path)
        return 0, 0, '', image_counter


executor = ThreadPoolExecutor(max_workers=16)


dimensions = {}
for subset in subsets:

    subset_path = os.path.join(dataset_path, subset)
    subset_images_path = os.path.join(subset_path, 'IMAGES')

    images = [image for image in os.listdir(subset_images_path) if 'jpg' in image or 'png' in image]

    futures = []

    image_counter = 0
    for image in tqdm.tqdm(images):
        image_counter += 1

        image_path = os.path.join(subset_images_path, image)

        futures.append(executor.submit(get_dimension, image_path, image_counter))

    for future in concurrent.futures.as_completed(futures):
        height, width, image_path, counter = future.result()

        print('{}/{}'.format(counter,image_counter))

        dimension_label = '{}_{}'.format(height, width)

        if dimension_label not in dimensions:

            dimensions[dimension_label] = [1, image_path]

        else:

            counter = dimensions[dimension_label][0]
            counter += 1

            dimensions[dimension_label][0] = counter

print(dimensions)
