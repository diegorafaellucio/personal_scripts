import os
import shutil

import cv2
import imutils
import tqdm

images_path = '/home/diego/images_video'

images = sorted(os.listdir(images_path))

for image in tqdm.tqdm(images):

    image_path = os.path.join(images_path, image)

    img = cv2.imread(image_path)
    rotated_image = imutils.rotate_bound(img, 90)

    cv2.imwrite(image_path, rotated_image)
    # cv2.imshow('image', imutils.resize(rotated_image, height=800))
    # cv2.waitKey(0)

    # shutil.move(image_path, new_image_path)