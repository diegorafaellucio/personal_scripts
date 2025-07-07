#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def parse_dataset_path(image_path):
    """
    Parse the dataset path from an image path.
    
    Example:
    /home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0/IMAGES/...
    
    Returns:
    - base_path: Path up to ECOTRACE/BARRA_MANSA
    - dataset_name: BM
    - dataset_version: 9.0
    - image_name: filename
    """
    try:
        # Find the second occurrence of "ECOTRACE"
        ecotrace_pos = image_path.find("ECOTRACE")
        if ecotrace_pos >= 0:
            second_ecotrace_pos = image_path.find("ECOTRACE", ecotrace_pos + 1)
            if second_ecotrace_pos >= 0:
                # Split the path at the second ECOTRACE
                parts = image_path.split('/', second_ecotrace_pos)
                
                # Extract the relevant parts
                path_parts = image_path.split('/')
                
                # Find indices for IMAGES, dataset name and version
                for i, part in enumerate(path_parts):
                    if part == "IMAGES":
                        images_index = i
                        dataset_version_index = i - 1
                        dataset_name_index = i - 2
                        break
                else:
                    # If IMAGES not found
                    return None, None, None, None
                
                # Extract the components
                dataset_name = path_parts[dataset_name_index]
                dataset_version = path_parts[dataset_version_index]
                image_name = path_parts[-1]
                
                # Construct the base path
                base_path = '/'.join(path_parts[:dataset_name_index])
                
                return base_path, dataset_name, dataset_version, image_name
    except Exception as e:
        print(f"Error parsing path: {e}")
    
    return None, None, None, None


def get_annotation_path(base_path, dataset_name, dataset_version, image_name):
    """
    Convert an image path to its corresponding annotation path.
    """
    # Construct the path to the annotations directory
    annotation_dir = os.path.join(base_path, dataset_name, dataset_version, 'ANNOTATIONS')
    
    # Change the file extension from .jpg to .xml
    annotation_file = os.path.splitext(image_name)[0] + '.xml'
    
    return os.path.join(annotation_dir, annotation_file)


def count_annotations_by_class(xml_path):
    """
    Parse XML annotation file and count objects by class.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        class_counts = Counter()
        
        # Find all object elements
        for obj in root.findall('.//object'):
            # Get the class name
            class_name = obj.find('name').text if obj.find('name') is not None else 'unknown'
            class_counts[class_name] += 1
            
        return class_counts
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return Counter()


def analyze_dataset_files(input_path):
    """
    Analyze train.txt, test.txt, and eval.txt files to count annotations by class.
    """
    input_path = Path(input_path)
    
    # Check if the input path exists
    if not input_path.exists():
        print(f"Error: Path {input_path} does not exist.")
        return
    
    # Define the files to process
    files_to_process = ['train.txt', 'test.txt', 'eval.txt']
    
    # Dictionary to store dataset information
    dataset_info = {}
    
    # Global counters for the general summary
    global_stats = {
        'total_images': 0,
        'class_counts': Counter(),
        'split_counts': {'train': 0, 'test': 0, 'eval': 0},
        'datasets_count': 0,
        'annotations_per_dataset': {},
        'images_per_dataset': {}
    }
    
    # Process each file
    for filename in files_to_process:
        file_path = input_path / filename
        
        if not file_path.exists():
            print(f"Warning: File {filename} not found in {input_path}")
            continue
        
        print(f"Processing {filename}...")
        
        # Read the file
        with open(file_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        # Process each image path
        for image_path in image_paths:
            base_path, dataset_name, dataset_version, image_name = parse_dataset_path(image_path)
            
            if not all([base_path, dataset_name, dataset_version, image_name]):
                print(f"Warning: Could not parse path: {image_path}")
                continue
            
            # Create a unique key for this dataset
            dataset_key = f"{dataset_name}/{dataset_version}"
            
            # Initialize dataset info if not already present
            if dataset_key not in dataset_info:
                dataset_info[dataset_key] = {
                    'base_path': base_path,
                    'dataset_name': dataset_name,
                    'dataset_version': dataset_version,
                    'image_count': 0,
                    'class_counts': Counter(),
                    'split_counts': {'train': 0, 'test': 0, 'eval': 0}
                }
                global_stats['datasets_count'] += 1
            
            # Update image count
            dataset_info[dataset_key]['image_count'] += 1
            global_stats['total_images'] += 1
            
            # Update split count
            split_name = filename.split('.')[0]  # 'train', 'test', or 'eval'
            dataset_info[dataset_key]['split_counts'][split_name] += 1
            global_stats['split_counts'][split_name] += 1
            
            # Get annotation path
            annotation_path = get_annotation_path(base_path, dataset_name, dataset_version, image_name)
            
            # Count annotations by class
            if os.path.exists(annotation_path):
                class_counts = count_annotations_by_class(annotation_path)
                dataset_info[dataset_key]['class_counts'] += class_counts
                global_stats['class_counts'] += class_counts
    
    # Calculate additional global statistics
    for dataset_key, info in dataset_info.items():
        total_annotations = sum(info['class_counts'].values())
        global_stats['annotations_per_dataset'][dataset_key] = total_annotations
        global_stats['images_per_dataset'][dataset_key] = info['image_count']
    
    # Generate report
    generate_report(dataset_info, global_stats, input_path)
    
    # Generate visualizations
    generate_visualizations(dataset_info, global_stats, input_path)


def generate_visualizations(dataset_info, global_stats, input_path):
    """
    Generate visualizations for the dataset analysis.
    """
    output_dir = input_path / 'analysis_visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # 1. Class distribution pie chart
    plt.figure(figsize=(12, 8))
    labels = []
    sizes = []
    
    # Get the top 10 classes by count
    top_classes = global_stats['class_counts'].most_common(10)
    other_count = sum(count for cls, count in global_stats['class_counts'].items() 
                      if cls not in [c for c, _ in top_classes])
    
    for cls, count in top_classes:
        labels.append(cls)
        sizes.append(count)
    
    if other_count > 0:
        labels.append('Other')
        sizes.append(other_count)
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Classes (Top 10)')
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Split distribution bar chart
    plt.figure(figsize=(10, 6))
    splits = list(global_stats['split_counts'].keys())
    counts = list(global_stats['split_counts'].values())
    
    plt.bar(splits, counts, color=['blue', 'orange', 'green'])
    plt.title('Distribution of Images by Split')
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    for i, v in enumerate(counts):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    plt.savefig(output_dir / 'split_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dataset comparison bar chart (top 5 datasets by image count)
    plt.figure(figsize=(12, 8))
    
    # Sort datasets by image count
    sorted_datasets = sorted(global_stats['images_per_dataset'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
    
    dataset_names = [name for name, _ in sorted_datasets]
    image_counts = [count for _, count in sorted_datasets]
    
    plt.bar(dataset_names, image_counts, color='skyblue')
    plt.title('Top 5 Datasets by Image Count')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(image_counts):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_datasets.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def generate_report(dataset_info, global_stats, input_path):
    """
    Generate a report with dataset information and class counts.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = input_path / f'dataset_analysis_report_{timestamp}.txt'
    
    with open(output_path, 'w') as f:
        # Write the general summary (totalizador geral)
        f.write("=" * 80 + "\n")
        f.write("                      TOTALIZADOR GERAL DOS DATASETS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write(f"Total de Datasets: {global_stats['datasets_count']}\n")
        f.write(f"Total de Imagens: {global_stats['total_images']}\n\n")
        
        f.write("Distribuição por Split:\n")
        f.write(f"  - Train: {global_stats['split_counts']['train']} ({global_stats['split_counts']['train']/global_stats['total_images']*100:.1f}%)\n")
        f.write(f"  - Test: {global_stats['split_counts']['test']} ({global_stats['split_counts']['test']/global_stats['total_images']*100:.1f}%)\n")
        f.write(f"  - Eval: {global_stats['split_counts']['eval']} ({global_stats['split_counts']['eval']/global_stats['total_images']*100:.1f}%)\n\n")
        
        f.write("Contagem Total de Classes:\n")
        total_annotations = sum(global_stats['class_counts'].values())
        f.write(f"Total de Anotações: {total_annotations}\n")
        f.write(f"Média de Anotações por Imagem: {total_annotations/global_stats['total_images']:.2f}\n\n")
        
        f.write("Distribuição de Classes:\n")
        for class_name, count in sorted(global_stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_annotations) * 100
            f.write(f"  - {class_name}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("                      DETALHAMENTO POR DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        # Write individual dataset information
        for dataset_key, info in sorted(dataset_info.items(), key=lambda x: x[1]['image_count'], reverse=True):
            f.write(f"Dataset: {dataset_key}\n")
            f.write(f"Base Path: {info['base_path']}\n")
            f.write(f"Total Images: {info['image_count']} ({info['image_count']/global_stats['total_images']*100:.1f}% do total)\n")
            
            f.write(f"Split Distribution:\n")
            for split, count in info['split_counts'].items():
                if count > 0:
                    percentage = (count / info['image_count']) * 100
                    f.write(f"  - {split.capitalize()}: {count} ({percentage:.1f}%)\n")
            
            f.write("\nAnnotation Class Counts:\n")
            total_dataset_annotations = sum(info['class_counts'].values())
            
            if total_dataset_annotations > 0:
                f.write(f"Total Annotations: {total_dataset_annotations}\n")
                f.write(f"Average Annotations per Image: {total_dataset_annotations/info['image_count']:.2f}\n\n")
                
                for class_name, count in sorted(info['class_counts'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_dataset_annotations) * 100
                    global_percentage = (count / total_annotations) * 100
                    f.write(f"  - {class_name}: {count} ({percentage:.1f}% do dataset, {global_percentage:.1f}% do total)\n")
            else:
                f.write("  No annotations found\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training data files and generate a comprehensive summary of all datasets.')
    parser.add_argument('--input_path', help='Path to directory containing train.txt, test.txt, and eval.txt files', 
                        default='/home/diego/2TB/TREINOS/BARRAMANSA_4.0+LP_1.0/DATA')
    
    args = parser.parse_args()
    
    print(f"Analyzing datasets in: {args.input_path}")
    analyze_dataset_files(args.input_path)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
