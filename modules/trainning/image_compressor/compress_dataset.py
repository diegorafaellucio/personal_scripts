import os
import cv2
import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def change_image_quality(image_path, image_counter):
    img = cv2.imread(image_path)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return image_counter


def compress_images(datataset_path, images_folder='IMAGES'):
    subsets = os.listdir(datataset_path)

    for subset in subsets:
        subset_path = os.path.join(datataset_path, subset)

        subset_images_path = os.path.join(subset_path, images_folder)

        executor = ThreadPoolExecutor(max_workers=512)

        futures = []
        files = os.listdir(subset_images_path)
        images = [file for file in files if 'jpg' in file]
        image_counter = 0
        for image in tqdm.tqdm(images):
            image_counter += 1
            image_path = os.path.join(subset_images_path, image)
            futures.append(executor.submit(change_image_quality, image_path, image_counter))

        finished_counter = 0
        for _ in concurrent.futures.as_completed(futures):
            # pass
            finished_counter += 1
            progress = '{}/{}'.format(finished_counter, len(images))
            print(progress)


if __name__ == '__main__':
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/GERAL/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/GERAL/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/GERAL/4.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/BARRA_MANSA/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/ARN/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/ARN/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/ARN/3.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BLN/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BLN/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/2.1')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/3.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/3.1')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/3.2')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/JBO/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/3.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/4.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/4.1')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PGO/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PGO/1.1')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PGO/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PGO/2.1')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PRN/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/2.1')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/3.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/4.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/4.1')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/ARN/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/BTS/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/JBO/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/JNB/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/MSO/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/MSO/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/PGO/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/PRN/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/AUDITORIA/MINERVA/RLM/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/RIO_MARIA/RM/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/SUL_BEEF/SB/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/1.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/2.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/3.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/4.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/5.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/6.0')
    # compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/7.0')
    compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/8-GREASE/TRAIN/ECOTRACE/MINERVA/BLN/1.0', images_folder='NO_BACKGROUND_IMAGES')
    compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/8-GREASE/TRAIN/ECOTRACE/MINERVA/BLN/2.0', images_folder='NO_BACKGROUND_IMAGES')
    compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/8-GREASE/TRAIN/ECOTRACE/MINERVA/BLN/3.0', images_folder='NO_BACKGROUND_IMAGES')
    compress_images('/home/diego/2TB/datasets/GCP/eco/bovinos/8-GREASE/TRAIN/ECOTRACE/MINERVA/BLN/4.0', images_folder='NO_BACKGROUND_IMAGES')






