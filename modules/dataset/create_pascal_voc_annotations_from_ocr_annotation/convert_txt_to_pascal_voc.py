#!/usr/bin/env python3
"""
Script to convert OCR text annotations to Pascal VOC XML format.

Handles two different text annotation templates:
1. Template with camera info and vehicle position (class: brazilian_plate)
2. Template with layout info (class: brazilian_plate or mercosur_plate based on layout)

Author: Diego Rafael Lucio
Date: 2025-08-12
"""

import os
import re
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
from typing import Dict, List, Tuple, Optional


def parse_coordinates(coord_str: str) -> List[Tuple[int, int]]:
    """
    Parse coordinate string and return list of (x, y) tuples.
    
    Args:
        coord_str: String containing coordinates like "441,425 570,424 572,468 444,467"
    
    Returns:
        List of (x, y) coordinate tuples
    """
    coords = []
    coord_pairs = coord_str.strip().split()
    
    for pair in coord_pairs:
        if ',' in pair:
            x, y = map(int, pair.split(','))
            coords.append((x, y))
    
    return coords


def get_bbox_from_coords(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box from coordinate list.
    
    Args:
        coords: List of (x, y) coordinate tuples
    
    Returns:
        Tuple of (xmin, ymin, xmax, ymax)
    """
    if not coords:
        return (0, 0, 0, 0)
    
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    
    return (xmin, ymin, xmax, ymax)


def parse_template_1(content: str) -> Optional[Dict]:
    """
    Parse first template format (with camera info and vehicle position).
    
    Args:
        content: Text content of the annotation file
    
    Returns:
        Dictionary with parsed data or None if not this template
    """
    lines = content.strip().split('\n')
    
    # Check if this is template 1 by looking for camera info
    if not any(line.startswith('camera:') for line in lines):
        return None
    
    data = {'class_name': 'brazilian_plate'}
    
    for line in lines:
        line = line.strip()
        if line.startswith('corners:'):
            coord_str = line.replace('corners:', '').strip()
            coords = parse_coordinates(coord_str)
            if coords:
                data['bbox'] = get_bbox_from_coords(coords)
                data['coordinates'] = coords
            break
    
    return data if 'bbox' in data else None


def parse_template_2(content: str) -> Optional[Dict]:
    """
    Parse second template format (with layout info).
    
    Args:
        content: Text content of the annotation file
    
    Returns:
        Dictionary with parsed data or None if not this template
    """
    lines = content.strip().split('\n')
    
    # Check if this is template 2 by looking for layout info
    layout_line = None
    corners_line = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('layout:'):
            layout_line = line
        elif line.startswith('corners:'):
            corners_line = line
    
    if not layout_line or not corners_line:
        return None
    
    # Determine class name based on layout
    layout = layout_line.replace('layout:', '').strip().lower()
    if 'brazilian' in layout:
        class_name = 'brazilian_plate'
    elif 'mercosur' in layout:
        class_name = 'mercosur_plate'
    else:
        class_name = 'brazilian_plate'  # default
    
    # Parse coordinates
    coord_str = corners_line.replace('corners:', '').strip()
    coords = parse_coordinates(coord_str)
    
    if not coords:
        return None
    
    data = {
        'class_name': class_name,
        'bbox': get_bbox_from_coords(coords),
        'coordinates': coords
    }
    
    return data


def parse_annotation_file(file_path: str) -> Optional[Dict]:
    """
    Parse annotation file and return extracted data.
    
    Args:
        file_path: Path to the annotation text file
    
    Returns:
        Dictionary with parsed data or None if parsing failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try template 1 first
        data = parse_template_1(content)
        if data:
            return data
        
        # Try template 2
        data = parse_template_2(content)
        if data:
            return data
        
        print(f"Warning: Could not parse file {file_path}")
        return None
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def create_pascal_voc_xml(image_filename: str, image_width: int, image_height: int, 
                         objects: List[Dict]) -> ElementTree:
    """
    Create Pascal VOC XML annotation.
    
    Args:
        image_filename: Name of the image file
        image_width: Width of the image
        image_height: Height of the image
        objects: List of object dictionaries with bbox and class info
    
    Returns:
        ElementTree object containing the XML
    """
    annotation = Element('annotation')
    
    # Folder
    folder = SubElement(annotation, 'folder')
    folder.text = 'images'
    
    # Filename
    filename = SubElement(annotation, 'filename')
    filename.text = image_filename
    
    # Path
    path = SubElement(annotation, 'path')
    path.text = f'images/{image_filename}'
    
    # Source
    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    
    # Size
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(image_width)
    height = SubElement(size, 'height')
    height.text = str(image_height)
    depth = SubElement(size, 'depth')
    depth.text = '3'
    
    # Segmented
    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # Objects
    for obj in objects:
        object_elem = SubElement(annotation, 'object')
        
        name = SubElement(object_elem, 'name')
        name.text = obj['class_name']
        
        pose = SubElement(object_elem, 'pose')
        pose.text = 'Unspecified'
        
        truncated = SubElement(object_elem, 'truncated')
        truncated.text = '0'
        
        difficult = SubElement(object_elem, 'difficult')
        difficult.text = '0'
        
        bndbox = SubElement(object_elem, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(obj['bbox'][0])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(obj['bbox'][1])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(obj['bbox'][2])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(obj['bbox'][3])
    
    return ElementTree(annotation)


def process_annotations(input_dir: str, output_dir: str, default_width: int = 1920, 
                       default_height: int = 1080):
    """
    Process all annotation files in the input directory.
    
    Args:
        input_dir: Directory containing text annotation files
        output_dir: Directory to save Pascal VOC XML files
        default_width: Default image width if not specified
        default_height: Default image height if not specified
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Get all .txt files
    txt_files = list(input_path.glob('*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Processing {len(txt_files)} annotation files...")
    
    processed = 0
    failed = 0
    
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        
        # Parse annotation
        data = parse_annotation_file(str(txt_file))
        
        if not data:
            print(f"Failed to parse: {txt_file.name}")
            failed += 1
            continue
        
        # Create corresponding image filename (assuming same name with different extension)
        image_filename = txt_file.stem + '.jpg'  # You can modify this as needed
        
        # Create Pascal VOC XML
        objects = [data]  # Single object per file based on your examples
        xml_tree = create_pascal_voc_xml(image_filename, default_width, default_height, objects)
        
        # Save XML file
        xml_filename = txt_file.stem + '.xml'
        xml_path = output_path / xml_filename
        
        try:
            xml_tree.write(str(xml_path), encoding='utf-8', xml_declaration=True)
            processed += 1
            print(f"Created: {xml_filename}")
        except Exception as e:
            print(f"Error saving {xml_filename}: {e}")
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed} files")
    print(f"Failed: {failed} files")


def main():
    """Main function to run the conversion."""
    input_dir = "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_TXT/"
    output_dir = "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_XML/"
    
    print("OCR Text to Pascal VOC XML Converter")
    print("=" * 40)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    process_annotations(input_dir, output_dir)


if __name__ == "__main__":
    main()
