#!/usr/bin/env python3
"""
analyze_labels.py

This script analyzes JSON label files in a directory and counts the occurrences of each label.
It displays a summary of the results, showing how many times each label appears across all files.

Usage:
    python analyze_labels.py [directory_path]

If no directory path is provided, the script will prompt the user to enter one.
"""

import os
import json
import sys
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from tqdm import tqdm


def analyze_directory(directory_path, recursive=False):
    """
    Analyze all JSON files in the specified directory and count label occurrences.
    
    Args:
        directory_path (str): Path to the directory containing JSON files
        recursive (bool): Whether to search subdirectories recursively
        
    Returns:
        tuple: (total_label_counts, file_counts, files_with_label, total_files, images_info)
            - total_label_counts: Counter of all label occurrences
            - file_counts: dict mapping labels to number of files containing that label
            - files_with_label: dict mapping labels to list of files containing that label
            - total_files: total number of JSON files processed
            - images_info: dict with information about images and their labels
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return Counter(), {}, {}, 0, {}
    
    total_label_counts = Counter()
    file_counts = defaultdict(int)
    files_with_label = defaultdict(list)
    total_files = 0
    images_info = {}
    
    print(f"Analyzing JSON files in {directory_path}...")
    
    # Get all JSON files in the directory
    if recursive:
        json_files = list(Path(directory_path).rglob('*.json'))
    else:
        json_files = list(Path(directory_path).glob('*.json'))
    
    # Use tqdm for a progress bar
    for file_path in tqdm(json_files, desc="Processing JSON files"):
        filename = file_path.name
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check if the file has the expected structure
            if 'shapes' not in data:
                print(f"Warning: {filename} does not have 'shapes' field, skipping")
                continue
                
            # Count labels in this file
            file_labels = Counter()
            for shape in data['shapes']:
                if 'label' in shape:
                    label = shape['label']
                    file_labels[label] += 1
            
            # Update the total counts
            total_label_counts.update(file_labels)
            
            # Update file counts for each label
            for label, count in file_labels.items():
                file_counts[label] += 1
                files_with_label[label].append(str(file_path))
            
            # Store image info
            image_name = data.get('imagePath', filename.replace('.json', ''))
            images_info[str(file_path)] = {
                'image_name': image_name,
                'labels': dict(file_labels),
                'total_annotations': sum(file_labels.values())
            }
                
            total_files += 1
                
        except json.JSONDecodeError:
            print(f"Warning: {filename} is not a valid JSON file, skipping")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return total_label_counts, file_counts, files_with_label, total_files, images_info


def display_results(total_label_counts, file_counts, files_with_label, total_files):
    """
    Display the analysis results in a formatted way.
    
    Args:
        total_label_counts (Counter): Counter of all label occurrences
        file_counts (dict): Dict mapping labels to number of files containing that label
        files_with_label (dict): Dict mapping labels to list of files containing that label
        total_files (int): Total number of JSON files processed
    """
    if not total_label_counts:
        print("No labels found in the specified directory.")
        return
        
    print("\n" + "="*80)
    print(f"LABEL ANALYSIS SUMMARY")
    print(f"Total JSON files processed: {total_files}")
    print("="*80)
    
    # Sort labels by frequency (most common first)
    sorted_labels = sorted(total_label_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate the maximum label length for formatting
    max_label_length = max(len(label) for label, _ in sorted_labels)
    
    # Print header
    print(f"{'LABEL':<{max_label_length+5}} | {'COUNT':<10} | {'FILES':<10} | {'AVG PER FILE':<15} | {'% OF TOTAL':<15}")
    print("-"*(max_label_length+5) + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*15 + "-+-" + "-"*15)
    
    total_annotations = sum(total_label_counts.values())
    
    # Print each label's statistics
    for label, count in sorted_labels:
        files = file_counts[label]
        avg_per_file = count / files if files > 0 else 0
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        print(f"{label:<{max_label_length+5}} | {count:<10} | {files:<10} | {avg_per_file:.2f}{' '*11} | {percentage:.2f}%")
    
    print("="*80)
    print(f"Total annotations: {total_annotations}")
    print("="*80)


def main(directory_path=None, recursive=False):
    """
    Main function to run the label analysis.
    
    Args:
        directory_path (str, optional): Directory containing JSON files to analyze
        recursive (bool, optional): Whether to search subdirectories recursively
    """
    # If no directory provided, use command line args or prompt the user
    if directory_path is None:
        parser = argparse.ArgumentParser(description='Analyze JSON label files in a directory')
        parser.add_argument('directory', nargs='?', help='Directory containing JSON files to analyze')
        parser.add_argument('-r', '--recursive', action='store_true', help='Search subdirectories recursively')
        args = parser.parse_args()
        
        directory_path = args.directory
        recursive = args.recursive
        
        # If still no directory provided, prompt the user
        if not directory_path:
            directory_path = input("Enter the directory path containing JSON files: ")
    
    # Analyze the directory
    total_label_counts, file_counts, files_with_label, total_files, _ = analyze_directory(
        directory_path, recursive)
    
    # Display the results
    display_results(total_label_counts, file_counts, files_with_label, total_files)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main(directory_path="/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0/ANNOTATIONS_JSON")
