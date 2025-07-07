#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
import re
from pathlib import Path
import cv2
import numpy as np

def validate_yolo_label(label_path, image_path, class_count=4):
    """
    Validate a YOLO format label file to ensure all values are within expected ranges
    and classes match the expected class count.
    
    Args:
        label_path: Path to the YOLO label file
        image_path: Path to the corresponding image file
        class_count: Number of valid classes (0 to class_count-1)
        
    Returns:
        (is_valid, error_message): Tuple with validation status and error message if invalid
    """
    # Check if label file exists
    if not os.path.exists(label_path):
        return False, f"Label file missing: {label_path}"
    
    # Check if image file exists
    if not os.path.exists(image_path):
        return False, f"Image file missing: {image_path}"
    
    try:
        # Read image dimensions
        img = cv2.imread(image_path)
        if img is None:
            return False, f"Cannot read image file: {image_path}"
        
        img_height, img_width = img.shape[:2]
        
        # Read label content
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            # Empty label file may be valid if there are no objects
            return True, "Empty label file"
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                return False, f"Invalid format in line {line_num}: {line} (expected 5 values)"
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Check class ID is valid
                if class_id < 0 or class_id >= class_count:
                    return False, f"Invalid class ID {class_id} in line {line_num} (must be 0-{class_count-1})"
                
                # Check coordinates are normalized (0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    return False, f"Coordinates out of bounds in line {line_num}: {line} (must be 0-1)"
                
                # Check for very small or zero width/height
                if width < 0.001 or height < 0.001:
                    return False, f"Box too small in line {line_num}: width={width}, height={height}"
                
                # Verify the bounding box is within the image
                x1 = (x_center - width/2)
                y1 = (y_center - height/2)
                x2 = (x_center + width/2)
                y2 = (y_center + height/2)
                
                if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                    return False, f"Box extends outside image in line {line_num}: {line}"
                    
            except ValueError:
                return False, f"Invalid numeric values in line {line_num}: {line}"
                
        return True, "Valid label file"
        
    except Exception as e:
        return False, f"Error validating label: {str(e)}"

def check_dataset(dataset_path, class_count=4):
    """
    Check all label files in the dataset for validity.
    
    Args:
        dataset_path: Path to the dataset data file (train.txt, test.txt, eval.txt)
        class_count: Number of valid classes (0 to class_count-1)
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return
    
    print(f"Checking labels in dataset: {dataset_path}")
    
    # Statistics
    total_images = 0
    valid_labels = 0
    invalid_labels = 0
    missing_labels = 0
    errors_by_type = {}
    
    # Read the dataset file
    with open(dataset_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    for image_path in image_paths:
        total_images += 1
        
        # Get corresponding label path
        label_path = os.path.splitext(image_path)[0] + '.txt'
        
        # Validate label
        is_valid, error_message = validate_yolo_label(label_path, image_path, class_count)
        
        if not is_valid:
            invalid_labels += 1
            if "missing" in error_message.lower():
                missing_labels += 1
                
            # Count error types
            error_type = error_message.split(':')[0]
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            
            # Print detailed error message
            print(f"ERROR: {error_message}")
            if "Invalid class ID" in error_message:
                # Show the problematic label content
                try:
                    with open(label_path, 'r') as f:
                        content = f.read()
                    print(f"Label content: {content.strip()}")
                except:
                    pass
        else:
            valid_labels += 1
            
    # Print summary
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Total images: {total_images}")
    print(f"Valid labels: {valid_labels} ({valid_labels/total_images*100:.1f}%)")
    print(f"Invalid labels: {invalid_labels} ({invalid_labels/total_images*100:.1f}%)")
    print(f"Missing labels: {missing_labels} ({missing_labels/total_images*100:.1f}%)")
    
    if errors_by_type:
        print("\nError types:")
        for error_type, count in sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count} ({count/invalid_labels*100:.1f}%)")
    
    print("=" * 60)
    
    return invalid_labels == 0

def fix_invalid_labels(dataset_path, class_count=4):
    """
    Attempt to fix invalid label files by ensuring classes are within range
    and coordinates are properly bounded.
    
    Args:
        dataset_path: Path to the dataset data file (train.txt, test.txt, eval.txt)
        class_count: Number of valid classes (0 to class_count-1)
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return
    
    print(f"Fixing invalid labels in dataset: {dataset_path}")
    fixed_count = 0
    
    # Read the dataset file
    with open(dataset_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    for image_path in image_paths:
        # Get corresponding label path
        label_path = os.path.splitext(image_path)[0] + '.txt'
        
        # Check if label file exists
        if not os.path.exists(label_path):
            # Create empty label file
            with open(label_path, 'w') as f:
                pass
            print(f"Created empty label file: {label_path}")
            fixed_count += 1
            continue
            
        # Read label content
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        modified = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                # Invalid format, skip this line
                modified = True
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Fix class ID if needed
                if class_id < 0 or class_id >= class_count:
                    # Skip this annotation if class ID is invalid
                    modified = True
                    continue
                
                # Bound coordinates to 0-1 range
                x_center = max(0.001, min(0.999, x_center))
                y_center = max(0.001, min(0.999, y_center))
                width = max(0.001, min(0.999, width))
                height = max(0.001, min(0.999, height))
                
                # Ensure box stays within image
                half_width = width / 2
                half_height = height / 2
                
                if x_center - half_width < 0:
                    x_center = half_width
                    modified = True
                    
                if x_center + half_width > 1:
                    x_center = 1 - half_width
                    modified = True
                    
                if y_center - half_height < 0:
                    y_center = half_height
                    modified = True
                    
                if y_center + half_height > 1:
                    y_center = 1 - half_height
                    modified = True
                
                fixed_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                fixed_lines.append(fixed_line)
                
            except ValueError:
                # Invalid values, skip this line
                modified = True
                continue
                
        if modified:
            # Write fixed content back to the file
            with open(label_path, 'w') as f:
                for line in fixed_lines:
                    f.write(f"{line}\n")
            print(f"Fixed label file: {label_path}")
            fixed_count += 1
    
    print(f"Fixed {fixed_count} label files.")
    return fixed_count

def main():
    parser = argparse.ArgumentParser(description='Check and fix YOLO format label files')
    parser.add_argument('--dataset_path', type=str, default='/home/diego/2TB/TREINOS/BARRAMANSA_4.0/DATA',
                       help='Path to directory containing train.txt, test.txt, and eval.txt files')
    parser.add_argument('--class_count', type=int, default=4, 
                       help='Number of valid classes (0 to class_count-1)')
    parser.add_argument('--fix', action='store_true',
                       help='Attempt to fix invalid label files')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific file to check (train.txt, test.txt, or eval.txt)')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if args.file:
        files_to_check = [dataset_path / args.file]
    else:
        files_to_check = [
            dataset_path / 'train.txt',
            dataset_path / 'test.txt',
            dataset_path / 'eval.txt'
        ]
    
    all_valid = True
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"Warning: File {file_path} not found, skipping.")
            continue
            
        print(f"\nChecking {file_path.name}...")
        is_valid = check_dataset(str(file_path), args.class_count)
        all_valid = all_valid and is_valid
        
        if not is_valid and args.fix:
            print(f"\nFixing {file_path.name}...")
            fix_invalid_labels(str(file_path), args.class_count)
    
    if all_valid:
        print("\nAll label files are valid!")
    else:
        if not args.fix:
            print("\nSome label files have issues. Run with --fix to attempt automatic fixes.")
        else:
            print("\nFixed invalid labels. Please run check again to verify.")

if __name__ == "__main__":
    main()
