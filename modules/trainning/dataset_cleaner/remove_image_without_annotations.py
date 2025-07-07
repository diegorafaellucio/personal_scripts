#!/usr/bin/env python3
"""
remove_image_without_annotations.py

This script identifies images that don't have corresponding annotation files
and moves them to a 'NOT_ANNOTATED_IMAGES' directory instead of deleting them.

Usage:
    Import and call the main function with appropriate parameters.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def move_unannotated_images(images_dir, annotations_dir, dry_run=False):
    """
    Move images without corresponding annotation files to a NOT_ANNOTATED_IMAGES directory.
    
    Args:
        images_dir (str): Directory containing image files
        annotations_dir (str): Directory containing annotation files
        dry_run (bool): If True, only print what would be moved without actually moving
        
    Returns:
        int: Number of images moved
    """
    # Validate directories
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images directory does not exist: {images_dir}")
    if not os.path.isdir(annotations_dir):
        raise ValueError(f"Annotations directory does not exist: {annotations_dir}")
    
    # Create target directory for unannotated images
    not_annotated_dir = os.path.join(os.path.dirname(images_dir), "NOT_ANNOTATED_IMAGES")
    if not dry_run and not os.path.exists(not_annotated_dir):
        os.makedirs(not_annotated_dir)
        print(f"Created directory: {not_annotated_dir}")
    
    # Get list of files
    image_files = os.listdir(images_dir)
    annotation_files = [os.path.splitext(file)[0] for file in os.listdir(annotations_dir)]
    
    moved_images_count = 0
    
    print(f"Checking {len(image_files)} images against {len(annotation_files)} annotation files...")
    
    # Process each image file
    for image_file in tqdm(image_files, desc="Processing images"):
        image_base_name = os.path.splitext(image_file)[0]
        
        # Check if image has a corresponding annotation
        if image_base_name not in annotation_files:
            source_path = os.path.join(images_dir, image_file)
            target_path = os.path.join(not_annotated_dir, image_file)
            
            if dry_run:
                print(f"Would move: {source_path} -> {target_path}")
            else:
                try:
                    shutil.move(source_path, target_path)
                    print(f"Moved: {image_file}")
                except Exception as e:
                    print(f"Error moving {image_file}: {str(e)}")
                    continue
                    
            moved_images_count += 1
    
    return moved_images_count


def main(images_dir=None, annotations_dir=None, dry_run=False):
    """
    Main function to move unannotated images.
    
    Args:
        images_dir (str, optional): Directory containing image files
        annotations_dir (str, optional): Directory containing annotation files
        dry_run (bool, optional): If True, only print what would be moved without actually moving
        
    Returns:
        int: Number of images moved
    """
    # Default paths
    default_images_dir = '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/BARRA_MANSA/BM/4.0/AUSENTE/IMAGES'
    default_annotations_dir = '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/BARRA_MANSA/BM/4.0/AUSENTE/ANNOTATIONS'
    
    # Use provided paths or defaults
    images_dir = images_dir if images_dir else default_images_dir
    annotations_dir = annotations_dir if annotations_dir else default_annotations_dir
    
    # Move unannotated images
    try:
        moved_count = move_unannotated_images(images_dir, annotations_dir, dry_run)
        
        if dry_run:
            print(f"\nDRY RUN: Would move {moved_count} unannotated images")
        else:
            print(f"\nSuccessfully moved {moved_count} unannotated images to NOT_ANNOTATED_IMAGES")
        
        return moved_count
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 0


if __name__ == "__main__":
    # Use default paths when running as script
    main(images_dir="/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/SULBEEF/SB/1.0/IMAGES", annotations_dir="/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/SULBEEF/SB/1.0/ANNOTATIONS")
