#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to analyze YOLO format TXT annotations and count occurrences of each label.
"""

import os
import argparse
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_yolo_labels(directory_path, label_map=None):
    """
    Analyze YOLO format TXT annotations and count occurrences of each label.
    
    Args:
        directory_path: Path to the directory containing YOLO TXT annotations
        label_map: Dictionary mapping class indices to label names
    
    Returns:
        Counter object with label counts
    """
    if label_map is None:
        # Default label map for the bruise dataset
        label_map = {
            0: "FALHA",
            1: "LEVE",
            2: "MODERADA",
            3: "GRAVE",
            4: "GRAVE_ABCESSO"
        }
    
    # Initialize counter for labels
    label_counter = Counter()
    
    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    print(f"Found {total_files} annotation files to analyze")
    
    # Process each file
    for i, txt_file in enumerate(txt_files, 1):
        file_path = os.path.join(directory_path, txt_file)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        print(f"Warning: Invalid format in {txt_file}, line: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        label = label_map.get(class_id, f"class_{class_id}")
                        label_counter[label] += 1
                    except ValueError:
                        print(f"Warning: Invalid class ID in {txt_file}, line: {line}")
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
        
        # Print progress
        if i % 100 == 0 or i == total_files:
            print(f"Processed {i}/{total_files} files")
    
    return label_counter


def plot_label_distribution(label_counter, output_path=None):
    """
    Plot the distribution of labels.
    
    Args:
        label_counter: Counter object with label counts
        output_path: Path to save the plot image
    """
    labels = list(label_counter.keys())
    counts = list(label_counter.values())
    
    # Sort by count (descending)
    sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color='skyblue')
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha='center',
            va='bottom'
        )
    
    plt.title('Distribution of Labels in YOLO Annotations')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    plt.show()


def main(directory_path=None):
    """
    Main function to analyze YOLO labels.
    
    Args:
        directory_path: Path to the directory containing YOLO TXT annotations
    """
    parser = argparse.ArgumentParser(description='Analyze YOLO format TXT annotations')
    parser.add_argument('--dir', type=str, help='Directory containing YOLO TXT annotations')
    parser.add_argument('--output', type=str, help='Path to save the plot image')
    
    args = parser.parse_args()
    
    # Use provided directory path or command line argument
    directory_path = directory_path or args.dir
    
    if not directory_path:
        print("Error: Directory path not provided")
        return
    
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return
    
    # Label map for the bruise dataset
    label_map = {
        0: "FALHA",
        1: "LEVE",
        2: "MODERADA",
        3: "GRAVE",
        4: "GRAVE_ABCESSO"
    }
    
    # Analyze labels
    label_counter = analyze_yolo_labels(directory_path, label_map)
    
    # Print results
    print("\nLabel Distribution:")
    for label, count in label_counter.most_common():
        print(f"{label}: {count}")
    
    # Calculate total annotations
    total_annotations = sum(label_counter.values())
    print(f"\nTotal annotations: {total_annotations}")
    
    # Calculate percentage for each label
    print("\nLabel Percentages:")
    for label, count in label_counter.most_common():
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        print(f"{label}: {percentage:.2f}%")
    
    # Plot distribution
    output_path = args.output or os.path.join(os.path.dirname(directory_path), "label_distribution.png")
    plot_label_distribution(label_counter, output_path)


if __name__ == "__main__":
    main(directory_path="/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0/ANNOTATIONS_TXT")
