import time

import  cv2
import numpy as np
import torch

def get_segmented_objects_countour(image, masks):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h = image.shape[0]
    original_w = image.shape[1]

    start = time.time()

    masks = np.array(masks.cpu())
    stop = time.time()
    # print(stop-start)
    contour_all = []
    for i, mask in enumerate(masks):
        annotation = mask.astype(np.uint8)

        mask_size = np.sum(annotation == 1)
        # print(mask_size)



        # cv2.imshow('test', annotation)
        # cv2.waitKey(0)

        if mask_size > 15000:

            annotation = cv2.resize(
                annotation,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )

            contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)


    return contour_all