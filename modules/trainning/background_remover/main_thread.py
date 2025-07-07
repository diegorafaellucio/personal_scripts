import os.path
import shutil

import concurrent.futures
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import imutils
import tqdm

from fastsam import FastSAM
from shapely import Polygon
import cv2
import numpy as np

import gc
import torch

from utils.mask_tools import get_segmented_objects_countour


model = FastSAM('weights/FastSAM-x.pt')
DEVICE = 'cuda'

def get_best_intersection_score_result(padding, image, contour_coords):
    image_height, image_width = image.shape[:2]


    gt_x_min = padding
    gt_y_min = 0
    gt_x_max = image_width - padding
    gt_y_max = image_height

    padding_polygon = Polygon(
        [(gt_x_min, gt_y_min), (gt_x_min, gt_y_max), (gt_x_max, gt_y_max), (gt_x_max, gt_y_min)])


    contour_result_x_min = contour_coords[0]
    contour_result_y_min = contour_coords[1]
    contour_result_x_max = contour_coords[2]
    contour_result_y_max = contour_coords[3]

    side_polygon = Polygon(
        [(contour_result_x_min, contour_result_y_min), (contour_result_x_min, contour_result_y_max),
         (contour_result_x_max, contour_result_y_max),
         (contour_result_x_max, contour_result_y_min)])

    intersection = padding_polygon.intersection(side_polygon)

    intersection_score = intersection.area / padding_polygon.area

    return intersection_score

def generate_image(image_path, image_output_path, not_possible_to_remove_background_image_output_path, counter):
    print(counter)
    try:
        if not os.path.exists(image_output_path):
            image = cv2.imread(image_path)
            everything_results = model.predict(image, device=DEVICE, retina_masks=False, imgsz=480, conf=0.4, iou=0.5, verbose=False )
            start = time.time()
            masks = everything_results[0].masks.data
            stop = time.time()
            elapsed = stop -start
            # print(elapsed)

            start_2 = time.time()
            contour_all = get_segmented_objects_countour(image, masks)
            stop_2 = time.time()
            elapsed_2 = stop_2 - start_2
            # print(elapsed_2)

            best_contour = None
            best_contour_score = 0

            padding = int(image.shape[1]/3)

            start_3 = time.time()
            for contour in contour_all:
                x, y, w, h = cv2.boundingRect(contour)
                intersection_score = get_best_intersection_score_result(padding, image, (x, y, x + w, y + h))

                if intersection_score > 0.20:
                    if intersection_score > best_contour_score:
                        best_contour_score = intersection_score
                        best_contour = contour

            stencil = np.zeros(image.shape).astype(image.dtype)
            stencil = cv2.fillPoly(stencil, pts=[best_contour], color=(255, 255, 255))

            stop_3 = time.time()
            elapsed_3 = stop_3 - start_3
            # print(elapsed_3)

            # print(elapsed+elapsed_2+elapsed_3)



            output_image = cv2.bitwise_and(image, stencil)

            # bgr_output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_output_path, output_image)


            torch.cuda.empty_cache()
            gc.collect()


            return counter
    except Exception as ex:
        print(traceback.format_exc())
        # generate_image(image_path, image_output_path, not_possible_to_remove_background_image_output_path, counter)
        time.sleep(1)
        shutil.copy(image_path, not_possible_to_remove_background_image_output_path)
        return counter

def generate_images_without_brackground(dataset_path):
    executor = ThreadPoolExecutor(max_workers=1)
    futures = []


    images_with_brackground_path = os.path.join(dataset_path, 'IMAGES')
    images_without_brackground_path = os.path.join(dataset_path, 'NO_BACKGROUND_IMAGES')
    not_possible_to_remove_brackground_path = os.path.join(dataset_path, 'NOT_POSSIBLE_TO_REMOVE_BACKGROUND')

    if not os.path.exists(images_without_brackground_path):
        # shutil.rmtree(images_without_brackground_path)
        os.mkdir(images_without_brackground_path)
    # else:
    #     os.mkdir(images_without_brackground_path)


    if os.path.exists(not_possible_to_remove_brackground_path):
        shutil.rmtree(not_possible_to_remove_brackground_path)
        os.mkdir(not_possible_to_remove_brackground_path)
    else:
        os.mkdir(not_possible_to_remove_brackground_path)

    input_images  = [input_image for input_image in os.listdir(images_with_brackground_path) if 'jpg' in input_image or 'png' in input_image ]


    for image_counter, input_image in enumerate(input_images):
        if image_counter < 10:
            time.sleep(1)
        # else:
            # time.sleep(0.3)
        log = '{}/{}'.format(image_counter+1, len(input_images))
        print(log)
        image_path = os.path.join(images_with_brackground_path, input_image)
        image_output_path = os.path.join(images_without_brackground_path, input_image)
        not_possible_to_remove_background_image_output_path = os.path.join(not_possible_to_remove_brackground_path,
                                                                           input_image)

        # generate_image(image_path, image_output_path, not_possible_to_remove_background_image_output_path, log)
        futures.append(executor.submit(generate_image, image_path, image_output_path, not_possible_to_remove_background_image_output_path, log))

    amount_counter = 0
    for future in concurrent.futures.as_completed(futures):
        logger = future.result()
        amount_counter +=1
        #
        # print('{}/{}'.format(amount_counter, len(images)))



if __name__ == '__main__':
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/EXCESSIVA')
    #
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/EXCESSIVA')
    #
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/EXCESSIVA')
    #
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/EXCESSIVA')
    #
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/EXCESSIVA')
    #
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/EXCESSIVA')

    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/EXCESSIVA')

    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/AUSENTE')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/EXCESSIVA')

    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/AUDITORIA_BOVINOS/MINERVA/MSO/2024/02/13')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/8-COR_DE_GORDURA/MINERVA/BELEN/1.0/AMARILLO')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/AUDITORIA_BOVINOS/MINERVA/BLN/2024/01/31/')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/MINERVA/ARN/2.0/ESCASSA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/MINERVA/ARN/2.0/MEDIANA')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/MINERVA/ARN/2.0/UNIFORME')
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/MINERVA/ARN/2.0/EXCESSIVA')
