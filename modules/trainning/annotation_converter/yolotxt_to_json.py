#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert YOLO txt annotations to JSON format compatible with LabelMe.
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image


def get_image_dimensions(image_path):
    """Get the width and height of an image."""
    with Image.open(image_path) as img:
        return img.width, img.height


def convert_yolo_to_json(txt_file_path, image_file_path, label_map):
    """
    Convert YOLO txt annotation to JSON format.
    
    Args:
        txt_file_path: Path to the YOLO txt annotation file
        image_file_path: Path to the corresponding image file
        label_map: Dictionary mapping class indices to label names
    
    Returns:
        Dictionary containing the JSON annotation
    """
    # Get image dimensions
    try:
        img_width, img_height = get_image_dimensions(image_file_path)
    except Exception as e:
        print(f"Error reading image {image_file_path}: {e}")
        return None
    
    # Create JSON structure
    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": f"../IMAGES/{os.path.basename(image_file_path)}",  # Use relative path with ../IMAGES/ prefix
        "imageData": None,  # Add imageData field with None value
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    # Read YOLO annotation file
    try:
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading annotation file {txt_file_path}: {e}")
        return None
    
    # Process each bounding box
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) != 5:
            print(f"Invalid format in {txt_file_path}, line: {line}")
            continue
            
        class_id, x_center, y_center, width, height = parts
        
        try:
            class_id = int(class_id)
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)
        except ValueError as e:
            print(f"Error parsing values in {txt_file_path}, line: {line}, error: {e}")
            continue
        
        # Convert normalized YOLO coordinates to absolute pixel coordinates
        # Keep as floating point values instead of converting to integers
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height
        
        # Ensure coordinates are within image boundaries
        x1 = max(0.0, min(x1, img_width - 1))
        y1 = max(0.0, min(y1, img_height - 1))
        x2 = max(0.0, min(x2, img_width - 1))
        y2 = max(0.0, min(y2, img_height - 1))
        
        # Get label name from class ID
        label = label_map.get(class_id, f"class_{class_id}")
        
        # Create shape object with correct format for LabelMe
        shape = {
            "label": label,
            "points": [
                [x1, y1],
                [x2, y2]
            ],
            "group_id": None,
            "description": None,  # Add description field as in reference JSON
            "shape_type": "rectangle",
            "flags": {}
        }
        
        json_data["shapes"].append(shape)
    
    return json_data


def process_directory(txt_dir, img_dir, output_dir, label_map):
    """
    Process all txt files in a directory and convert them to JSON.
    
    Args:
        txt_dir: Directory containing YOLO txt annotation files
        img_dir: Directory containing corresponding image files
        output_dir: Directory to save JSON annotation files
        label_map: Dictionary mapping class indices to label names
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all txt files
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    print(f"Found {total_files} annotation files to process")
    
    success_count = 0
    error_count = 0
    
    for i, txt_file in enumerate(txt_files, 1):
        base_name = os.path.splitext(txt_file)[0]
        
        # Find corresponding image file
        img_file = f"{base_name}.jpg"
        img_path = os.path.join(img_dir, img_file)
        
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            error_count += 1
            continue
        
        # Convert annotation
        json_data = convert_yolo_to_json(
            os.path.join(txt_dir, txt_file),
            img_path,
            label_map
        )
        
        if json_data:
            # Save JSON file
            json_file = f"{base_name}.json"
            json_path = os.path.join(output_dir, json_file)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            success_count += 1
        else:
            error_count += 1
        
        # Print progress
        if i % 100 == 0 or i == total_files:
            print(f"Processed {i}/{total_files} files")
    
    print(f"Conversion completed: {success_count} successful, {error_count} failed")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO txt annotations to JSON format')
    parser.add_argument('--txt_dir', type=str, required=True, help='Directory containing YOLO txt annotation files')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing image files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save JSON annotation files')
    
    args = parser.parse_args()
    
    # Label map according to the provided indices
    label_map = {
        0: "FALHA",
        1: "LEVE",
        2: "MODERADA",
        3: "GRAVE",
        4: "GRAVE_ABCESSO"
    }
    
    process_directory(args.txt_dir, args.img_dir, args.output_dir, label_map)


if __name__ == "__main__":
    main()
