import os
import shutil

import cv2
import tqdm

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from base.src.ultralytics_classifier.classifier import Classifier
from base.src.utils.utils import get_intersection_score, generate_json
import json
device = 'cuda:0'

padding = 550

classifier = Classifier('/home/diego/2TB/projects/ECO/ecoia-classifier/data/models/skeleton/weight.pt')
print()
# side_classifier = Classifier('/home/diego/1TB/ecotrace/frameworks/yolov5/runs/train/exp4/weights/last.pt', device=device, max_det=10)


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
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_MEDIANA/ESCASSA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_MEDIANA/MEDIANA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_MEDIANA/UNIFORME')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/RIO_MARIA_MEDIANA/ESCASSA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/RIO_MARIA_MEDIANA/MEDIANA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/RIO_MARIA_MEDIANA/UNIFORME')

    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_MEDIANA/ESCASSA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_MEDIANA/MEDIANA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_UNIFORME_COMPARTILHAMENTO/EXCESSIVA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_UNIFORME_COMPARTILHAMENTO/MEDIANA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/3-MEAT/ECOTRACE/SUL_BEEF_UNIFORME_COMPARTILHAMENTO/UNIFORME')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/AUDITORIA/MINERVA/NOVA_AUDITORIA_MSO/AUSENTE')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/AUDITORIA/MINERVA/NOVA_AUDITORIA_MSO/ESCASSA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/AUDITORIA/MINERVA/NOVA_AUDITORIA_MSO/MEDIANA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/AUDITORIA/MINERVA/NOVA_AUDITORIA_MSO/UNIFORME')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/AUDITORIA/MINERVA/NOVA_AUDITORIA_MSO/EXCESSIVA')

    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/SUL_BEEF/1.0/AUSENTE')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/SUL_BEEF/1.0/ESCASSA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/SUL_BEEF/1.0/MEDIANA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/SUL_BEEF/1.0/UNIFORME')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/SUL_BEEF/1.0/EXCESSIVA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/AUSENTE')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/ESCASSA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/MEDIANA')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/UNIFORME')
    # generate_jsons('/home/diego/2TB/datasets/eco/BOVINOS/DATASETS/NOVOS/3-MEAT/ECOTRACE/RIO_MARIA/1.0/EXCESSIVA')

    generate_jsons('/home/diego/Downloads/Acabamento_novo_BM_3/AUSENTE', "AUSENTE")
    generate_jsons('/home/diego/Downloads/Acabamento_novo_BM_3/ESCASSA', "ESCASSA")
    generate_jsons('/home/diego/Downloads/Acabamento_novo_BM_3/MEDIANA', "MEDIANA")
    generate_jsons('/home/diego/Downloads/Acabamento_novo_BM_3/UNIFORME', "UNIFORME")
    # generate_jsons('/home/diego/Downloads/Acabamento_novo_BM_3/EXCESSIVA', "EXCESSIVA")

