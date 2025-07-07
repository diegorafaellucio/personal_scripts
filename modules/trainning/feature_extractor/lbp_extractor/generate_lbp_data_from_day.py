import os
import shutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

matlab_script_path = '/home/diego/wwn-0x5000039851d01007-part1/diego/Mestrado/Codes/Xeno_Canto_Songs'
matlab_script = 'matlab -nodesktop -nosplash -r "lbp_dataset_bovino_by_image(\'{}\', \'{}\', \'{}\', {}, {} );exit();"'

IMAGES = 'NO_BACKGROUND_IMAGES'
LBP_DATA = 'LBP_DATA'
# WHITE_LIST = ["UNIFORME"]

LBP_NEIGHBORS = 8
LBP_DISTANCE = 2



def process_image(class_id, image_path, lbp_path, distance, neighbors):

    command = matlab_script.format(class_id, image_path, lbp_path, distance, neighbors)

    try:

        stream = os.popen(command)
        stream.read()

        return True

    except:
        return False




def process_dataset(dataset_path):
    executor = ThreadPoolExecutor(max_workers=12)
    #

    futures = []
    finished_counter = 0
    images_path = os.path.join(dataset_path, IMAGES)
    lbp_data_path = os.path.join(dataset_path, LBP_DATA)

    if not os.path.exists(lbp_data_path):
        os.mkdir(lbp_data_path)
    # else:
    #     continue
    #     shutil.rmtree(lbp_data_path)
    #     os.mkdir(lbp_data_path)

    images = [image for image in os.listdir(images_path) if 'jpg' in image or 'png' in image]

    for image_name in images:
        lbp_name = image_name.replace('jpg', 'lbp').replace('png', 'lbp')

        image_path = os.path.join(images_path, image_name)
        lbp_path = os.path.join(lbp_data_path, lbp_name)

        # process_image(class_id, image_path, lbp_path, LBP_DISTANCE, LBP_NEIGHBORS)
        futures.append(
            executor.submit(process_image, 0, image_path, lbp_path, LBP_DISTANCE, LBP_NEIGHBORS))

    for future in concurrent.futures.as_completed(futures):
        finished_counter += 1
        processed = future.result()

        if processed:
            print('{}: {}/{}'.format('dataset', finished_counter, len(images)))


if __name__ == '__main__':
    os.chdir(matlab_script_path)

    process_dataset('/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_BOVINOS/ARN/2024/01/16')
