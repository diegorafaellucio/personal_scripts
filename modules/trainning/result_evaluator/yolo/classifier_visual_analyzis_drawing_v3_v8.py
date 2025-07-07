import os
import shutil

import cv2
import tqdm

from base.src.utils.utils import get_intersection_score

from base.src.new_classifier.classifier import Classifier
from base.src.new_classifier.classifier_v8 import ClassifierV8
from shapely import Polygon
import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

colors = {
    'LEVE': (0, 255, 0),
    'MODERADA': (0, 165, 255),
    'GRAVE': (0, 255, 255),
    'FALHA': (255, 204, 51),
    'ETIQUETA': (110, 204, 51),
    'PLACA': (255, 50, 51)
}


def get_best_result(results):
    best_result = None
    best_score = 0

    for result in results:
        score = result['confidence']
        if score > best_score:
            best_result = result

    return best_result


def remove_if_intersection_with_stamp(bruise_detections, stamp_detections ):
    black_list = []
    for first_counter, first_detection in enumerate(bruise_detections):
        for second_counter, second_detection in enumerate(stamp_detections):
            first_detection_x_min = first_detection['topleft']['x']
            first_detection_y_min = first_detection['topleft']['y']
            first_detection_x_max = first_detection['bottomright']['x']
            first_detection_y_max = first_detection['bottomright']['y']

            gt_coords = [first_detection_x_min, first_detection_y_min, first_detection_x_max, first_detection_y_max]

            second_detection_x_min = second_detection['topleft']['x']
            second_detection_y_min = second_detection['topleft']['y']
            second_detection_x_max = second_detection['bottomright']['x']
            second_detection_y_max = second_detection['bottomright']['y']

            detection_coords = [second_detection_x_min, second_detection_y_min, second_detection_x_max,
                                second_detection_y_max]

            intersection_score = get_intersection_score(gt_coords, detection_coords)

            if intersection_score > 0.50:
                black_list.append(first_counter)

    black_list = sorted(black_list, reverse=True)

    for index_to_remove in black_list:
        del bruise_detections[index_to_remove]

    return bruise_detections


def remove_region_intersection(detections):
    black_list = []
    for first_counter, first_detection in enumerate(detections):
        for second_counter, second_detection in enumerate(detections):
            if (first_counter != second_counter):
                first_detection_x_min = first_detection['topleft']['x']
                first_detection_y_min = first_detection['topleft']['y']
                first_detection_x_max = first_detection['bottomright']['x']
                first_detection_y_max = first_detection['bottomright']['y']
                first_detection_condidence = first_detection['confidence']

                gt_coords = [first_detection_x_min, first_detection_y_min, first_detection_x_max, first_detection_y_max]

                second_detection_x_min = second_detection['topleft']['x']
                second_detection_y_min = second_detection['topleft']['y']
                second_detection_x_max = second_detection['bottomright']['x']
                second_detection_y_max = second_detection['bottomright']['y']
                second_detection_condidence = second_detection['confidence']

                detection_coords = [second_detection_x_min, second_detection_y_min, second_detection_x_max,
                                    second_detection_y_max]

                intersection_score = get_intersection_score(gt_coords, detection_coords)

                if intersection_score > 0.80:
                    if first_detection_condidence > second_detection_condidence:
                        if second_counter not in black_list:
                            black_list.append(second_counter)
                    else:
                        if first_counter not in black_list:
                            black_list.append(first_counter)

    black_list = sorted(black_list, reverse=True)

    for index_to_remove in black_list:
        del detections[index_to_remove]

    return detections


def get_intersection_score(gt_coords, side_coords):
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_coords

    padding_polygon = Polygon(
        [(gt_x_min, gt_y_min), (gt_x_min, gt_y_max), (gt_x_max, gt_y_max), (gt_x_max, gt_y_min)])

    side_x_min, side_y_min, side_x_max, side_y_max = side_coords

    side_polygon = Polygon(
        [(side_x_min, side_y_min), (side_x_min, side_y_max), (side_x_max, side_y_max),
         (side_x_max, side_y_min)])

    intersection = padding_polygon.intersection(side_polygon)

    if intersection.area == 0:
        intersection_score = 0
    else:
        intersection_score = intersection.area / padding_polygon.area

    return intersection_score

def draw_detections(image_path, detections):
    image = cv2.imread(image_path)

    for detection in detections:
        label = detection['label'].upper()
        confidence = detection['confidence']

        if confidence >= 0.20:

            x_min = detection['topleft']['x']
            y_min = detection['topleft']['y']
            x_max = detection['bottomright']['x']
            y_max = detection['bottomright']['y']

            width = x_max - x_min
            height = y_max - y_min



            area = round((width*height), 2)

            label = label.split('-')[-1]

            if label.upper() == 'FALHA' and area < 1400:
                continue

            area = str(area)
            to_print = '{} {}'.format(area,label, round(confidence, 2))


            color = colors[label]

            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            (w, h), _ = cv2.getTextSize(
                to_print, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Prints the text.
            image = cv2.rectangle(image, (x_min, y_min - 20), (x_min + w, y_min), color, -1)
            image = cv2.putText(image, to_print, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # # For printing text
            # image = cv2.putText(image, label, (x_min, y_min),
            #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


    return image


def process_data(images_input_path, image, bruise_classifier, stamp_classifier, images_output_path):
    image_path = os.path.join(images_input_path, image)

    img = cv2.imread(image_path)

    bruise_detection_results = bruise_classifier.detect(img)
    stamp_detection_results = stamp_classifier.detect(img)

    filtered_detections = remove_region_intersection(bruise_detection_results)

    if stamp_detection_results is not None:

        no_stamp_filtered = remove_if_intersection_with_stamp(filtered_detections, stamp_detection_results)

        drawed_image = draw_detections(image_path, no_stamp_filtered)

    else:
        drawed_image = draw_detections(image_path, filtered_detections)


    new_image_path = os.path.join(images_output_path, image)

    # cv2.imwrite(new_image_path, drawed_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    cv2.imwrite(new_image_path, drawed_image)


def process_model(train_name, network_name, planta, images_input_path, images_output_path, bruise_classifier_model_path, stamp_classifier_model_path, device='cuda:0'):
    bruise_classifier = ClassifierV8(bruise_classifier_model_path)
    stamp_classifier = ClassifierV8(stamp_classifier_model_path)

    if not os.path.exists(images_output_path):
        os.makedirs(images_output_path)
    else:
        shutil.rmtree(images_output_path)
        os.makedirs(images_output_path)

    images = os.listdir(images_input_path)

    futures = []

    executor = ThreadPoolExecutor(max_workers=32)



    for image in tqdm.tqdm(images):
        # process_data(images_input_path, image, bruise_classifier, stamp_classifier, images_output_path)
        futures.append(executor.submit(process_data, images_input_path, image, bruise_classifier, stamp_classifier, images_output_path))


    finished_counter = 0

    for _ in concurrent.futures.as_completed(futures):
        finished_counter += 1

        print('{} {} {}: {}/{}'.format(train_name, network_name, planta,finished_counter, len(images)))


if __name__ == '__main__':

    # treino_barra_mansa nano 416

    treino = 'RIO_MARIA+BARRA_MANSA'
    tamanho_imagem = '416'
    rede = 'V5_MEDIUM'
    bruise_model_path = '/home/diego/Downloads/yolov8m_lesoes/detect/train/weights/best.pt'
    stamp_model_path = '/home/diego/Downloads/train_yolos_carimbo_e_etiqueta/train/weights/best.pt'

    process_model(treino, rede, 'BARRA_MANSA_RISCO_DE_SANGUE', '/home/diego/2TB/datasets/eco/AUDITORIA_BOVINOS/BARRA_MANSA/2024/03/07',
                  'lp_teste',
                  bruise_model_path, stamp_model_path)
    # 
    # process_model(treino, rede, 'SUL_BEEF','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/SUL_BEEF/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/SUL_BEEF/2024/02/02'.format(treino,
    #                                                                                                     tamanho_imagem,

    #                                                                                                     rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'ARN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/ARN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BTS','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/BTS/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/BTS/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PGO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PGO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PGO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PRN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PRN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PRN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'JBO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/JBO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/JBO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/09',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/09'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/10',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/10'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # # treino_barra_mansa small 416
    # 
    # treino = 'RIO_MARIA+BARRA_MANSA'
    # tamanho_imagem = '416'
    # rede = 'V5_SMALL'
    # path_modelo = '/home/diego/2TB/yolo/new_v5/runs/train/lesao_rio_maria_v5_small_416/weights/best.pt'
    # 
    # process_model(treino, rede, 'RIO_MARIA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/RIO_MARIA/2024/01/12',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/RIO_MARIA/2024/01/12'.format(treino,
    #                                                                                                      tamanho_imagem,
    #                                                                                                      rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'SUL_BEEF','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/SUL_BEEF/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/SUL_BEEF/2024/02/02'.format(treino,
    #                                                                                                     tamanho_imagem,
    #                                                                                                     rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'ARN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/ARN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'RIO_MARIA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/BTS/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/BTS/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PGO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PGO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PGO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PRN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PRN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PRN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'JBO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/JBO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/JBO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/09',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/09'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/10',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/10'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # # medium
    # treino = 'RIO_MARIA+BARRA_MANSA'
    # tamanho_imagem = '416'
    # rede = 'V5_MEDIUM'
    # path_modelo = '/home/diego/2TB/yolo/new_v5/runs/train/lesao_rio_maria_v5_medium_416/weights/best.pt'
    # 
    # process_model(treino, rede, 'RIO_MARIA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/RIO_MARIA/2024/01/12',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/RIO_MARIA/2024/01/12'.format(treino,
    #                                                                                                      tamanho_imagem,
    #                                                                                                      rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'SUL_BEEF','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/SUL_BEEF/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/SUL_BEEF/2024/02/02'.format(treino,
    #                                                                                                     tamanho_imagem,
    #                                                                                                     rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'ARN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/ARN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BTS','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/BTS/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/BTS/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PGO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PGO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PGO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PRN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PRN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PRN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'JBO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/JBO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/JBO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/09',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/09'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/10',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/10'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # # large
    # 
    # # medium
    # treino = 'RIO_MARIA+BARRA_MANSA'
    # tamanho_imagem = '416'
    # rede = 'V5_LARGE'
    # path_modelo = '/home/diego/2TB/yolo/new_v5/runs/train/lesao_rio_maria_v5_large_416/weights/'
    # 
    # process_model(treino, rede, 'RIO_MARIA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/RIO_MARIA/2024/01/12',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/RIO_MARIA/2024/01/12'.format(treino,
    #                                                                                                      tamanho_imagem,
    #                                                                                                      rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'SUL_BEEF','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/SUL_BEEF/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/SUL_BEEF/2024/02/02'.format(treino,
    #                                                                                                     tamanho_imagem,
    #                                                                                                     rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'ARN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/ARN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BTS','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/BTS/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/BTS/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PGO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PGO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PGO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'PRN','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PRN/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PRN/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'JBO','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/JBO/2024/02/02',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/JBO/2024/02/02'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/09',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/09'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)
    # 
    # process_model(treino, rede, 'BARRA_MANSA','/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/10',
    #               '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/10'.format(treino,
    #                                                                                                        tamanho_imagem,
    #                                                                                                        rede),
    #               path_modelo)

    # # EXTtra_large
    # treino = 'RIO_MARIA+BARRA_MANSA'
    # tamanho_imagem = '416'
    # rede = 'V5_EXTRA_LARGE'
    # path_modelo = '/home/diego/2TB/yolo/new_v5/runs/train/lesao_rio_maria_v5_extra_large_416/weights/best.pt'
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/RIO_MARIA/2024/01/12',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/RIO_MARIA/2024/01/12'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/SUL_BEEF/2024/02/02',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/SUL_BEEF/2024/02/02'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/ARN/2024/02/02',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/ARN/2024/02/02'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/BTS/2024/02/02',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/BTS/2024/02/02'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PGO/2024/02/02',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PGO/2024/02/02'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/PRN/2024/02/02',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/PRN/2024/02/02'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/MINERVA/JBO/2024/02/02',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/MINERVA/JBO/2024/02/02'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/09',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/09'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
    #
    # process_model('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/BARRA_MANSA/2024/01/10',
    #                   '/home/diego/2TB/datasets/eco/BOVINOS/RESULTADOS/{}/{}/{}/BARRA_MANSA/2024/01/10'.format(treino, tamanho_imagem, rede),
    #                   path_modelo)
