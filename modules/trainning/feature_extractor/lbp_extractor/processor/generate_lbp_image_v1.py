import os
import cv2
import numpy as np
from modules.feature_descritptors.lbp.descriptor.lbp import LocalBinaryPatterns

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# WHITE_LIST = ["AUSENTE", "ESCASSA", "MEDIANA", "EXCESSIVA"]
WHITE_LIST = ["AUSENTE", "ESCASSA", "MEDIANA", "UNIFORME", "EXCESSIVA"]

def generate_lbp(image_path, feature_descriptor, output_image_path):

    try:
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        lbp_image = feature_descriptor.get_blp_image_v1(img_gray)
        lbp_image = lbp_image.astype(np.uint8)

        cv2.imwrite(output_image_path, lbp_image)

        return True
    except:
        return False


def describe_dataset(dataset_path, feature_descriptor, black_list=['DATA']):
    executor = ThreadPoolExecutor(max_workers=24)
    subsets = os.listdir(dataset_path)

    output = 'LBP_IMAGE_UNIFORM'

    for subset in subsets:
        futures = []
        finished_counter = 0


        if subset in WHITE_LIST:
            subset_path = os.path.join(dataset_path, subset)

            images_path = os.path.join(subset_path, 'NO_BACKGROUND_IMAGES')
            output_images_path = os.path.join(subset_path, output)

            if not os.path.exists(output_images_path):
                os.mkdir(output_images_path)

            images = [image for image in os.listdir(images_path) if 'jpg' in image or 'png' in image]

            for image in images:
                image_path = os.path.join(images_path, image)
                output_image_path = os.path.join(output_images_path, image)

                # generate_lbp(image_path, feature_descriptor, output_image_path)

                futures.append(
                    executor.submit(generate_lbp,image_path, feature_descriptor, output_image_path))




            for future in concurrent.futures.as_completed(futures):
                finished_counter += 1
                processed = future.result()

                if processed:
                    print('{}: {}/{}'.format(subset, finished_counter, len(images)))


if __name__ == '__main__':

    lbp_descriptor = LocalBinaryPatterns(8, 2)

    describe_dataset('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/MINERVA/ARN/1.0', lbp_descriptor)
