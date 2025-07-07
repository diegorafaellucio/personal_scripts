import os
import shutil

images_path = '/home/diego/images_video'

images = os.listdir(images_path)

for image in images:
    image_data = os.path.splitext(image)
    new_image = '{0:05d}'.format(int(image_data[0]))
    new_image = '{}{}'.format(new_image, image_data[1])

    image_path = os.path.join(images_path, image)
    new_image_path = os.path.join(images_path, new_image)

    shutil.move(image_path, new_image_path)