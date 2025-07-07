import cv2

image_path = '/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE/NO_BACKGROUND_IMAGES/20230506-0005-1-0110-1940.jpg'
default_image_path = '/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE/LBP_IMAGE/default-20230506-0005-1-0110-1940.jpg'
nri_uniform_image_path = '/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE/LBP_IMAGE/nri-uniform-20230506-0005-1-0110-1940.jpg'
ror_image_path = '/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE/LBP_IMAGE/ror-20230506-0005-1-0110-1940.jpg'
uniform_image_path = '/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE/LBP_IMAGE/uniform-20230506-0005-1-0110-1940.jpg'
var_image_path = '/home/diego/2TB/datasets/eco/BOVINOS/AUDITORIA_ARN/AUSENTE/LBP_IMAGE/var-20230506-0005-1-0110-1940.jpg'



image = cv2.imread(image_path)
default_image = cv2.imread(default_image_path)
nri_uniform_image = cv2.imread(nri_uniform_image_path)
ror_image = cv2.imread(ror_image_path)
uniform_image = cv2.imread(uniform_image_path)
var_image = cv2.imread(var_image_path)

output_image = cv2.hconcat([image, default_image, nri_uniform_image, ror_image, uniform_image, var_image])

# cv2.imshow('output',output_image)
# cv2.waitKey(0)

cv2.imwrite('sample.png', output_image)