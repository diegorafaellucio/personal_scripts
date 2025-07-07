import os.path
import shutil

import imutils
import tqdm

from fastsam import FastSAM, FastSAMPrompt
from shapely import Polygon
import cv2
import numpy as np


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

def generate_images_without_brackground(dataset_path):
    images_with_brackground_path = os.path.join(dataset_path, 'IMAGES')
    images_without_brackground_path = os.path.join(dataset_path, 'NO_BACKGROUND_IMAGES')
    not_possible_to_remove_brackground_path = os.path.join(dataset_path, 'NOT_POSSIBLE_TO_REMOVE_BACKGROUND')

    if os.path.exists(images_without_brackground_path):
        shutil.rmtree(images_without_brackground_path)
        os.mkdir(images_without_brackground_path)
    else:
        os.mkdir(images_without_brackground_path)

    images = [image for image in os.listdir(images_with_brackground_path) if 'jpg' in image or 'png' in image]

    for image in tqdm.tqdm(images):
        image_path = os.path.join(images_with_brackground_path, image)
        image_output_path = os.path.join(images_without_brackground_path, image)
        not_possible_to_remove_background_image_output_path = os.path.join(not_possible_to_remove_brackground_path,
                                                                           image)
        try:

            everything_results = model(image_path, device=DEVICE, retina_masks=True, imgsz=460, conf=0.5, iou=0.7,)
            prompt_process = FastSAMPrompt(image_path, everything_results, device=DEVICE)

            ann = prompt_process.everything_prompt()

            temp, contour_all = prompt_process.get_segmented_objects_countour(ann)

            best_contours = []

            for contour in contour_all:
                x, y, w, h = cv2.boundingRect(contour)
                intersection_score = get_best_intersection_score_result(500, prompt_process.img, (x, y, x + w, y+h))

                if intersection_score > 0.70:
                    best_contours.append(contour)

            stencil = np.zeros(prompt_process.img.shape).astype(prompt_process.img.dtype)
            stencil = cv2.fillPoly(stencil, pts=[best_contours[0]], color=(255, 255, 255))



            output_image = cv2.bitwise_and(prompt_process.img, stencil)

            bgr_output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(image_output_path, bgr_output_image)
        except:
            shutil.copy(image_path, not_possible_to_remove_background_image_output_path)



if __name__ == '__main__':
    # generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/EXCESSIVA')

    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BTS/EXCESSIVA')

    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JBO/EXCESSIVA')

    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_JNB/EXCESSIVA')

    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_MSO/EXCESSIVA')

    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PGO/EXCESSIVA')

    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_PRN/EXCESSIVA')

    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/AUSENTE')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/ESCASSA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/MEDIANA')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/UNIFORME')
    generate_images_without_brackground('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_RLM/EXCESSIVA')