from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# Define an inference source
source = '/home/diego/2TB/datasets/eco/BOVINOS/CAROL/AUSENTE/IMAGES/20210323-0037-1-0110-0791.jpg'

# Create a FastSAM model
model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

# Run inference on an image
everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# Prepare a Prompt Process object
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

# Everything prompt
ann = prompt_process.everything_prompt()

# Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
# Text prompt

# Point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
# ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
prompt_process.plot(annotations=ann, output='./')