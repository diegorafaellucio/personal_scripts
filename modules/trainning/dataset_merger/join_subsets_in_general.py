import os
import shutil

import tqdm

output = 'general'
BLACK_LIST = ['GENERAL']

def process_dataset(dataset_path):

    output_path = os.path.join(dataset_path, output)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    subsets = os.listdir(dataset_path)

    for subset in tqdm.tqdm(subsets):
        if subset not in BLACK_LIST:
            subset_path = os.path.join(dataset_path, subset)

            categories = os.listdir(subset_path)

            for category in tqdm.tqdm(categories):
                category_path = os.path.join(subset_path, category)

                category_images_path = os.path.join(category_path, 'IMAGES')

                category_output_path = os.path.join(output_path, category)

                category_images_output_path = os.path.join(category_output_path, 'IMAGES')

                if not os.path.exists(category_output_path):
                    os.mkdir(category_output_path)
                    os.mkdir(category_images_output_path)

                images = os.listdir(category_images_path)

                for image in tqdm.tqdm(images):
                    try:
                        category_image_path = os.path.join(category_images_path, image)
                        category_image_output_path = os.path.join(category_images_output_path, image)
                        shutil.copy(category_image_path, category_image_output_path)
                    except:
                        continue
if __name__ == '__main__':
    # process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/ARN')
    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/BTS')
    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/JBO')
    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/JNB')
    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/MSO')
    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/PGO')
    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/PRN')
    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/GT/RLM')