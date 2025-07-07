#!/usr/bin/env python3
"""
Script to concatenate train.txt, eval.txt, and test.txt files from subdirectories
in /home/diego/2TB/TREINOS into a single output folder BRUISE_TRAIN/data.
"""

import os
import shutil
from pathlib import Path

# Define source and destination paths
SOURCE_DIR = "/home/diego/2TB/TREINOS"
OUTPUT_DIR = "/home/diego/2TB/BRUISE_TRAIN_ABSCESSO/DATA"

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured directory exists: {directory}")

def concat_files(source_dirs, output_dir, filename):
    """
    Concatenate files with the same name from multiple directories into a single file.
    
    Args:
        source_dirs: List of source directories containing the files
        output_dir: Directory where the concatenated file will be saved
        filename: Name of the file to concatenate (e.g., 'train.txt')
    """
    output_file_path = os.path.join(output_dir, filename)
    
    # Create or truncate the output file
    with open(output_file_path, 'w') as output_file:
        total_lines = 0
        
        for source_dir in source_dirs:
            data_dir = os.path.join(source_dir, "DATA")
            source_file_path = os.path.join(data_dir, filename)
            
            if os.path.exists(source_file_path):
                print(f"Processing {source_file_path}")
                with open(source_file_path, 'r') as source_file:
                    file_content = source_file.read()
                    output_file.write(file_content)
                    
                    # Add a newline if the file doesn't end with one
                    if file_content and not file_content.endswith('\n'):
                        output_file.write('\n')
                    
                    # Count lines for reporting
                    lines = file_content.count('\n') + (0 if file_content.endswith('\n') else 1)
                    total_lines += lines
                    print(f"  - Added {lines} lines from {os.path.basename(source_dir)}")
            else:
                print(f"Warning: File {source_file_path} not found, skipping")
        
        print(f"Created {output_file_path} with {total_lines} total lines")


def main():
    # Ensure output directory exists
    ensure_dir_exists(OUTPUT_DIR)
    
    # Get all subdirectories in the source directory
    source_subdirs = [os.path.join(SOURCE_DIR, d) for d in os.listdir(SOURCE_DIR) 
                     if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not source_subdirs:
        print(f"No subdirectories found in {SOURCE_DIR}")
        return
    
    print(f"Found {len(source_subdirs)} subdirectories to process")
    
    # Concatenate each file type
    for filename in ["train.txt", "eval.txt", "test.txt"]:
        concat_files(source_subdirs, OUTPUT_DIR, filename)
    
    print("All files have been concatenated successfully!")

if __name__ == "__main__":
    main()
