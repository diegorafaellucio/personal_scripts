#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import torch


def process_model(input_dir, output_dir, model_path):
    """
    Process images using a YOLO model and save the results with bounding boxes, labels, and scores.
    
    Args:
        input_dir (str): Directory containing images to process
        output_dir (str): Directory to save processed images
        model_path (str): Path to the YOLO model weights
    """
    # Check if CUDA is available and set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Explicitly set the model to use GPU
    model.to(device)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'**/*{ext}')))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Convert BGR to RGB for model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference with GPU
        results = model(img_rgb, device=device)
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            # Draw bounding boxes, labels, and scores
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence score
                conf = float(box.conf[0])
                
                # Get class name
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label text with class name and confidence
                label = f"{cls_name}: {conf:.2f}"
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw text background
                cv2.rectangle(
                    img, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width + 10, y1), 
                    (0, 255, 0), 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    img,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
        
        # Save output image
        rel_path = os.path.relpath(img_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image with detections
        cv2.imwrite(output_path, img)
    
    print(f"Processing complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    # Print CUDA availability information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # # Call the process_model function with the specified paths
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/443',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3_BM9/trains/nano/640/runs/detect/train_nano_1.0_sgd/weights/best.pt')
    #
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/444',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3_BM9/trains/nano/640/runs/detect/train_small_1.0_sgd/weights/best.pt')

    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/445',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3_BM9/trains/nano/640/runs/detect/train_medium_1.0_sgd/weights/best.pt')

    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/446',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3/trains/nano/640/runs/detect/train_nano_1.0_sgd/weights/best.pt')
    #
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/447',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3/trains/nano/640/runs/detect/train_small_1.0_sgd/weights/best.pt')
    #
    #
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/448',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3/trains/nano/640/runs/detect/train_medium_1.0_sgd/weights/best.pt')
    #
    #
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/449',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_ABSCESSO_BM/trains/nano/640/runs/detect/train_nano_1.0_sgd/weights/best.pt')
    #
    #
    #
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/450',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_ABSCESSO_BM/trains/nano/640/runs/detect/train_small_1.0_sgd/weights/best.pt')
    #


    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/451',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_v3_BM9/trains/nano/640/runs/detect/train_medium_1.0_sgd/weights/best.pt')


    #
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/452',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_ABSCESSO/trains/nano/640/runs/detect/train_nano_1.0_sgd/weights/best.pt')
    #
    #
    #
    # process_model('/home/diego/2TB/validacao_treino_eco/BM',
    #               '/home/diego/2TB/validacao_treino_eco/resultados/453',
    #               '/home/diego/2TB/yolo/Trains/v8/object_detection/BRUISE/NEW_BRUISE_ABSCESSO/trains/nano/640/runs/detect/train_small_1.0_sgd/weights/best.pt')
    #
    #
    #
    process_model('/home/diego/2TB/validacao_treino_eco/bm_27_02/27',
                  '/home/diego/2TB/validacao_treino_eco/resultados/459',
                  '/home/diego/2TB/yolo/Trains/runs/ecotrace_meat_barramansa_4.0+lp_1.0_416_nano_sgd/weights/best.pt')
