import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

sam = sam_model_registry["vit_h"](checkpoint="/home/diego/Downloads/models/sam_vit_h_4b8939.pth")
device = "cuda"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam, output_mode='binary_mask')




image = cv2.imread('/home/diego/2TB/datasets/eco/BOVINOS/CAROL/AUSENTE/IMAGES/20210323-0037-1-0110-0791.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

print(masks)