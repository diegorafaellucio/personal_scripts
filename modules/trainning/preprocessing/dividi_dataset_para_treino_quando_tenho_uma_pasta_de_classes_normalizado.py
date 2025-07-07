#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import math
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import copy


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


def get_objects_in_annotation(xml_path):
    """
    Parse XML annotation file and return a list of object classes.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = []
        
        # Find all object elements
        for obj in root.findall('.//object'):
            # Get the class name
            class_name = obj.find('name').text if obj.find('name') is not None else 'unknown'
            objects.append(class_name)
            
        return objects
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []


def collect_dataset_information(dataset_paths):
    """
    Collect information about all datasets and their annotations.
    """
    # Dictionary to store all image paths and their annotations by class
    all_images = {}
    class_images = defaultdict(list)  # Maps class name to list of images containing that class
    class_counts = Counter()  # Total count of annotations by class
    
    # Track statistics per dataset
    dataset_stats = {}
    
    for dataset_path in dataset_paths:
        print(f"Processing dataset: {dataset_path}")
        images_path = os.path.join(dataset_path, 'IMAGES')
        
        if not os.path.exists(images_path):
            print(f"Warning: Images path does not exist: {images_path}")
            continue
        
        # Initialize stats for this dataset
        dataset_name = Path(dataset_path).name
        dataset_version = Path(dataset_path).parent.name
        dataset_key = f"{dataset_name}/{dataset_version}"
        
        dataset_stats[dataset_key] = {
            'path': dataset_path,
            'image_count': 0,
            'class_counts': Counter(),
            'images': []
        }
        
        # Get all image files
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            base_path, parsed_dataset_name, parsed_dataset_version, _ = parse_dataset_path(image_path)
            
            if not all([base_path, parsed_dataset_name, parsed_dataset_version]):
                # Use simplified approach for paths that don't match the expected format
                base_path = str(Path(dataset_path).parent)
                parsed_dataset_name = dataset_name
                parsed_dataset_version = dataset_version
            
            # Get annotation path
            annotation_path = get_annotation_path(base_path, parsed_dataset_name, parsed_dataset_version, image_file)
            
            if os.path.exists(annotation_path):
                # Get the objects in this annotation
                objects = get_objects_in_annotation(annotation_path)
                
                if objects:
                    # Store the image path and its annotations
                    all_images[image_path] = {
                        'annotation_path': annotation_path,
                        'objects': objects,
                        'dataset_key': dataset_key  # Track which dataset this image belongs to
                    }
                    
                    # Update dataset specific stats
                    dataset_stats[dataset_key]['image_count'] += 1
                    dataset_stats[dataset_key]['images'].append(image_path)
                    
                    # Update class counts for this dataset
                    for obj_class in objects:
                        dataset_stats[dataset_key]['class_counts'][obj_class] += 1
                    
                    # Update global class counts
                    for obj_class in objects:
                        class_counts[obj_class] += 1
                        class_images[obj_class].append(image_path)
    
    return all_images, class_images, class_counts, dataset_stats


def normalize_dataset(all_images, class_images, class_counts, dataset_stats):
    """
    Normalize the dataset based on the class with the fewest annotations.
    Returns a list of image paths that should be included in the normalized dataset.
    """
    # Find the class with the fewest annotations
    min_class = min(class_counts.items(), key=lambda x: x[1])
    min_class_name, min_count = min_class
    
    print(f"\nNormalizando dataset baseado na classe com menos amostras: {min_class_name} ({min_count} anotações)")
    print("Contagem de classes antes da normalização:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cls}: {count}")
    
    # Prepare normalized dataset
    normalized_images = set()
    normalized_class_counts = Counter()
    
    # Track which normalized images belong to which dataset
    normalized_dataset_stats = {key: {'image_count': 0, 'class_counts': Counter()} for key in dataset_stats.keys()}
    
    # For each class, randomly select up to 'min_count' images containing that class
    for class_name, images in class_images.items():
        # Shuffle the list of images containing this class
        shuffled_images = copy.copy(images)
        random.shuffle(shuffled_images)
        
        # Keep track of how many annotations of this class we've included
        included_count = 0
        
        # Try to include up to min_count annotations of this class
        for img_path in shuffled_images:
            if included_count >= min_count:
                break
                
            # Count how many instances of this class are in the image
            instances_in_image = all_images[img_path]['objects'].count(class_name)
            
            # If adding this image wouldn't exceed our target, include it
            if included_count + instances_in_image <= min_count:
                normalized_images.add(img_path)
                included_count += instances_in_image
                
                # Update normalized stats for the dataset this image belongs to
                dataset_key = all_images[img_path]['dataset_key']
                normalized_dataset_stats[dataset_key]['image_count'] += 1
    
    # Count the annotations in our normalized dataset
    for img_path in normalized_images:
        dataset_key = all_images[img_path]['dataset_key']
        
        for obj_class in all_images[img_path]['objects']:
            normalized_class_counts[obj_class] += 1
            normalized_dataset_stats[dataset_key]['class_counts'][obj_class] += 1
    
    print("\nContagem de classes após a normalização:")
    for cls, count in sorted(normalized_class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cls}: {count}")
    
    return list(normalized_images), normalized_class_counts, normalized_dataset_stats


def fileGenerator(data, file_name):
    """
    Write a list of image paths to a file.
    """
    with open(file_name, "w") as file:
        for path in data:
            file.write(f"{path}\n")


def generate_visualizations(class_counts, normalized_class_counts, output_path):
    """
    Generate visualizations comparing the original and normalized class distributions.
    """
    output_dir = Path(output_path) / 'analysis_visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # Bar chart comparing original vs normalized counts
    plt.figure(figsize=(14, 8))
    
    # Get all unique classes
    all_classes = sorted(set(list(class_counts.keys()) + list(normalized_class_counts.keys())))
    
    # Prepare data
    original_counts = [class_counts.get(cls, 0) for cls in all_classes]
    norm_counts = [normalized_class_counts.get(cls, 0) for cls in all_classes]
    
    # Set up bar positions
    x = np.arange(len(all_classes))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, original_counts, width, label='Original', color='skyblue')
    plt.bar(x + width/2, norm_counts, width, label='Normalizado', color='lightcoral')
    
    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Número de Anotações')
    plt.title('Comparação de Distribuição de Classes: Original vs. Normalizado')
    plt.xticks(x, all_classes, rotation=90)
    plt.legend()
    
    # Add some space at the bottom for the rotated labels
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / 'class_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original distribution
    labels = []
    sizes = []
    
    # Get the top classes by count
    top_classes = class_counts.most_common(10)
    other_count = sum(count for cls, count in class_counts.items() 
                      if cls not in [c for c, _ in top_classes])
    
    for cls, count in top_classes:
        labels.append(cls)
        sizes.append(count)
    
    if other_count > 0:
        labels.append('Other')
        sizes.append(other_count)
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Distribuição Original de Classes')
    
    # Normalized distribution
    labels = []
    sizes = []
    
    # Get the top classes by count
    top_classes = normalized_class_counts.most_common(10)
    other_count = sum(count for cls, count in normalized_class_counts.items() 
                      if cls not in [c for c, _ in top_classes])
    
    for cls, count in top_classes:
        labels.append(cls)
        sizes.append(count)
    
    if other_count > 0:
        labels.append('Other')
        sizes.append(other_count)
    
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Distribuição Normalizada de Classes')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def generateDatabase(dataset_paths, output_path):
    """
    Process datasets, normalize based on minimum class count, and split into train/test/eval.
    
    Args:
        dataset_paths: List of paths to dataset folders containing IMAGES and ANNOTATIONS
        output_path: Path where the output files will be saved
    """
    # Define split ratios
    eval_ratio = 0.1
    test_ratio = 0.2
    train_ratio = 0.7
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Collect information about all datasets
    print("\n=== Coletando informações dos datasets ===")
    all_images, class_images, class_counts, dataset_stats = collect_dataset_information(dataset_paths)
    
    if not all_images:
        print("Error: No valid images found in the datasets.")
        return
    
    print(f"\nTotal de imagens encontradas: {len(all_images)}")
    print(f"Total de classes encontradas: {len(class_counts)}")
    
    # Normalize the dataset
    print("\n=== Normalizando o dataset ===")
    normalized_images, normalized_class_counts, normalized_dataset_stats = normalize_dataset(
        all_images, class_images, class_counts, dataset_stats)
    
    if not normalized_images:
        print("Error: No images selected after normalization.")
        return
    
    print(f"\nTotal de imagens após normalização: {len(normalized_images)}")
    
    # Generate visualizations comparing original and normalized distributions
    generate_visualizations(class_counts, normalized_class_counts, output_path)
    
    # Shuffle the normalized images
    random.shuffle(normalized_images)
    
    # Calculate split counts
    total_images = len(normalized_images)
    eval_count = int(total_images * eval_ratio)
    test_count = int(total_images * test_ratio)
    train_count = total_images - eval_count - test_count
    
    # Split the dataset
    eval_images = normalized_images[:eval_count]
    test_images = normalized_images[eval_count:eval_count+test_count]
    train_images = normalized_images[eval_count+test_count:]
    
    # Define output file paths
    eval_file = os.path.join(output_path, "eval.txt")
    test_file = os.path.join(output_path, "test.txt")
    train_file = os.path.join(output_path, "train.txt")
    
    # Generate the files
    fileGenerator(eval_images, eval_file)
    fileGenerator(test_images, test_file)
    fileGenerator(train_images, train_file)
    
    print("\n=== Divisão do dataset ===")
    print(f"Train: {len(train_images)} imagens ({len(train_images)/total_images*100:.1f}%)")
    print(f"Test: {len(test_images)} imagens ({len(test_images)/total_images*100:.1f}%)")
    print(f"Eval: {len(eval_images)} imagens ({len(eval_images)/total_images*100:.1f}%)")
    
    # Generate a report
    generate_report(
        normalized_class_counts, 
        {
            'train': len(train_images),
            'test': len(test_images),
            'eval': len(eval_images)
        }, 
        output_path,
        dataset_paths,
        dataset_stats,
        normalized_dataset_stats
    )
    
    print(f"\nDataset normalizado gerado em: {output_path}")


def generate_report(class_counts, split_counts, output_path, dataset_paths, original_dataset_stats, normalized_dataset_stats):
    """
    Generate a report with normalized dataset information.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_path) / f'normalized_dataset_report_{timestamp}.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("                 RELATÓRIO DE DATASET NORMALIZADO\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        f.write("Datasets de origem:\n")
        for idx, path in enumerate(dataset_paths, 1):
            f.write(f"  {idx}. {path}\n")
        f.write("\n")
        
        f.write(f"Path de saída: {output_path}\n\n")
        
        f.write("Divisão do dataset:\n")
        total_images = sum(split_counts.values())
        for split, count in split_counts.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            f.write(f"  - {split.capitalize()}: {count} imagens ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Write global distribution information
        f.write("=" * 80 + "\n")
        f.write("                      TOTALIZADOR GERAL DOS DATASETS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Distribuição de Classes (Normalizada):\n")
        total_annotations = sum(class_counts.values())
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            f.write(f"  - {class_name}: {count} anotações ({percentage:.1f}%)\n")
        
        # Write individual dataset information
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("              INFORMAÇÕES DETALHADAS POR DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_key, stats in original_dataset_stats.items():
            f.write(f"Dataset: {dataset_key}\n")
            f.write(f"Path: {stats['path']}\n\n")
            
            # Original stats
            original_image_count = stats['image_count']
            original_annotation_count = sum(stats['class_counts'].values())
            f.write("ESTATÍSTICAS ORIGINAIS:\n")
            f.write(f"  Total de imagens: {original_image_count}\n")
            f.write(f"  Total de anotações: {original_annotation_count}\n")
            f.write(f"  Média de anotações por imagem: {original_annotation_count/original_image_count:.2f}\n\n")
            
            f.write("  Distribuição de Classes:\n")
            for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / original_annotation_count) * 100 if original_annotation_count > 0 else 0
                f.write(f"    - {class_name}: {count} anotações ({percentage:.1f}%)\n")
            
            # Normalized stats
            normalized_stats = normalized_dataset_stats.get(dataset_key, {})
            normalized_image_count = normalized_stats.get('image_count', 0)
            normalized_class_counts = normalized_stats.get('class_counts', Counter())
            normalized_annotation_count = sum(normalized_class_counts.values())
            
            f.write("\nESTATÍSTICAS APÓS NORMALIZAÇÃO:\n")
            f.write(f"  Total de imagens: {normalized_image_count}\n")
            f.write(f"  Total de anotações: {normalized_annotation_count}\n")
            
            if normalized_image_count > 0:
                f.write(f"  Média de anotações por imagem: {normalized_annotation_count/normalized_image_count:.2f}\n")
            
            f.write("\n  Distribuição de Classes (Normalizada):\n")
            for class_name, count in sorted(normalized_class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / normalized_annotation_count) * 100 if normalized_annotation_count > 0 else 0
                f.write(f"    - {class_name}: {count} anotações ({percentage:.1f}%)\n")
            
            # Calculate reduction percentage
            if original_image_count > 0:
                image_reduction = 100 - (normalized_image_count / original_image_count * 100)
                f.write(f"\n  Redução no número de imagens: {image_reduction:.1f}%\n")
            
            if original_annotation_count > 0:
                annotation_reduction = 100 - (normalized_annotation_count / original_annotation_count * 100)
                f.write(f"  Redução no número de anotações: {annotation_reduction:.1f}%\n")
            
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"Report generated: {output_file}")


def main():
    # parser = argparse.ArgumentParser(description='Generate a normalized dataset for training based on the class with the fewest samples.')
    # parser.add_argument('--datasets', nargs='+', help='List of paths to datasets',
    #                     default=['/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0/',
    #                              '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0/'])
    # parser.add_argument('--output', help='Path where the output files will be saved',
    #                     default='/home/diego/2TB/TREINOS/BM_9.0+PGO_1.0_NORMALIZADO/DATA/')
    #
    # args = parser.parse_args()
    #
    # print("=== Gerando Dataset Normalizado ===")
    # print(f"Datasets de origem: {args.datasets}")
    # print(f"Path de saída: {args.output}")
    
    generateDatabase(['/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0/',
                                 '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0/'], '/home/diego/2TB/TREINOS/BM_9.0+PGO_1.0_NORMALIZADO/DATA/')


if __name__ == "__main__":
    main()
