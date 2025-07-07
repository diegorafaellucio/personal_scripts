import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import os

import cv2
import imutils
import tqdm

#
# {'2592_1944': [43514, '/home/diego/2TB/datasets/eco/new/CLASSIFICATION/AUSENTE/IMAGES/20210323-0150-2-0110-0791.jpg'],
#  '1920_1080': [4873, '/home/diego/2TB/datasets/eco/new/CLASSIFICATION/AUSENTE/IMAGES/20210831-0047-2-0110-2583.jpg'],
#  '2592_1458': [2514, '/home/diego/2TB/datasets/eco/new/CLASSIFICATION/AUSENTE/IMAGES/20210503-0662-2-0110-2960.jpg'],
#  '2590_1942': [21594, '/home/diego/2TB/datasets/eco/new/CLASSIFICATION/AUSENTE/IMAGES/20220725-0134-2-0110-0421.jpg'],
#  '1919_1080': [1816, '/home/diego/2TB/datasets/eco/new/CLASSIFICATION/AUSENTE/IMAGES/20230119-0159-1-0110-2500.jpg'],
#  '1440_1080': [22, '/home/diego/2TB/datasets/eco/new/CLASSIFICATION/EXCESSIVA/IMAGES/20230118-0049-1-0110-2500.jpg'],
#  '1942_1080': [6, '/home/diego/2TB/datasets/eco/new/CLASSIFICATION/UNIFORME/IMAGES/20230118-0085-1-0110-2500.jpg']}

crop_and_resize_info = {
    '2592_1944': {'resize': True, 'resize_data': {'height': 1920, 'width': 0}, 'crop': True,
                  'crop_data': {'left': 180, 'top': 0, 'right': 180, 'bottom': 0}},

    '2592_1458': {'resize': True, 'resize_data': {'height': 1920, 'width': 0}, 'crop': False,
                  'crop_data': {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}},

    '2590_1942': {'resize': True, 'resize_data': {'height': 1920, 'width': 0}, 'crop': True,
                  'crop_data': {'left': 180, 'top': 0, 'right': 180, 'bottom': 0}},

    '1919_1080': {'resize': True, 'resize_data': {'height': 1920, 'width': 1080}, 'crop': False,
                  'crop_data': {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}},

    '1440_1080': {'resize': True, 'resize_data': {'height': 1920, 'width': 0}, 'crop': True,
                  'crop_data': {'left': 180, 'top': 0, 'right': 180, 'bottom': 0}},

    '1942_1080': {'resize': False, 'resize_data': {'height': 1920, 'width': 1080}, 'crop': True,
                  'crop_data': {'left': 0, 'top': 0, 'right': 0, 'bottom': 22}},

    '1920_1079': {'resize': True, 'resize_data': {'height': 1920, 'width': 1080}, 'crop': False,
                  'crop_data': {'left': 0, 'top': 0, 'right': 0, 'bottom': 22}},

    '1920_1180': {'resize': True, 'resize_data': {'height': 1920, 'width': 1080}, 'crop': True,
                  'crop_data': {'left': 50, 'top': 0, 'right': 50, 'bottom': 22}},

    '1898_980': {'resize': True, 'resize_data': {'height': 1920, 'width': 1080}, 'crop': True,
                 'crop_data': {'left': 0, 'top': 0, 'right': 0, 'bottom': 156}},

    '1764_1080': {'resize': True, 'resize_data': {'height': 1920, 'width': 1080}, 'crop': True,
                  'crop_data': {'left': 0, 'top': 0, 'right': 87, 'bottom': 0}},

    '1920_993': {'resize': True, 'resize_data': {'height': 1920, 'width': 1080}, 'crop': False,
                 'crop_data': {'left': 0, 'top': 0, 'right': 87, 'bottom': 0}},

}

dataset_path = '/home/diego/2TB/datasets/eco/BOVINOS/GT/MSO/general'
subsets = os.listdir(dataset_path)


def update_image(image_path, image_counter):
    try:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        dimension_label = '{}_{}'.format(height, width)

        if dimension_label == '1920_1079':
            print()

        if image_path == '/home/diego/2TB/datasets/eco/BOVINOS/GT/PRN/general/EXCESSIVA/IMAGES/20230118-0049-1-0110-2500.jpg':
            print()

        if dimension_label != '1920_1080':

            update_data = crop_and_resize_info[dimension_label]

            resize = update_data['resize']
            resize_data = update_data['resize_data']

            crop = update_data['crop']
            crop_data = update_data['crop_data']

            if resize:
                new_width = resize_data['width']
                new_height = resize_data['height']
                image = resize_image(image, new_width, new_height)

            if crop:
                left = crop_data['left']
                top = crop_data['top']
                right = crop_data['right']
                bottom = crop_data['bottom']

                image = crop_image(image, left, top, right, bottom)

            cv2.imwrite(image_path, image)
        return image_counter
    except:
        os.remove(image_path)
        return image_counter


def resize_image(image, width=0, height=0):
    if width != 0:
        dsize = (width, height)

        image = cv2.resize(image, dsize)
    else:
        image = imutils.resize(image, height=height)

    return image


def crop_image(image, left=0, top=0, right=0, bottom=0):
    height, width = image.shape[:2]
    image = image[top:height - bottom, left:width - right]

    return image


executor = ThreadPoolExecutor(max_workers=16)

dimensions = {}

WHITE_LIST = ["AUSENTE", "ESCASSA", "MEDIANA", "UNIFORME", "EXCESSIVA", 'INCLASSIFICAVEL']
for subset in subsets:

    if subset in WHITE_LIST:

        subset_path = os.path.join(dataset_path, subset)
        subset_images_path = os.path.join(subset_path, 'IMAGES')

        images = [image for image in os.listdir(subset_images_path) if 'jpg' in image or 'png' in image]

        futures = []

        image_counter = 0
        for image in tqdm.tqdm(images):
            image_counter += 1

            image_path = os.path.join(subset_images_path, image)

            futures.append(executor.submit(update_image, image_path, image_counter))
            # update_image(image_path)

        for future in concurrent.futures.as_completed(futures):
            counter = future.result()

            print('{}/{}'.format(counter, image_counter))
