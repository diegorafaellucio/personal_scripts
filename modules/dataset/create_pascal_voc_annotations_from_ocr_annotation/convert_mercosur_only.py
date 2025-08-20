#!/usr/bin/env python3
"""
Script to re-convert only Mercosur annotations to verify they're working correctly.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from convert_txt_to_pascal_voc import parse_annotation_file, create_pascal_voc_xml


def convert_mercosur_only():
    """Convert only Mercosur annotation files."""
    
    input_dir = "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_TXT/"
    output_dir = "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_XML_MERCOSUR_TEST/"
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all Mercosur files
    mercosur_files = []
    
    for txt_file in input_path.glob('*.txt'):
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
            if 'layout: Mercosur' in content:
                mercosur_files.append(txt_file)
        except:
            continue
    
    print(f"Found {len(mercosur_files)} Mercosur annotation files")
    
    processed = 0
    failed = 0
    
    for txt_file in mercosur_files[:10]:  # Process first 10 for testing
        print(f"\nProcessing: {txt_file.name}")
        
        # Parse annotation
        data = parse_annotation_file(str(txt_file))
        
        if not data:
            print(f"  Failed to parse!")
            failed += 1
            continue
        
        print(f"  Class: {data['class_name']}")
        print(f"  Bbox: {data['bbox']}")
        print(f"  Coords: {data.get('coordinates', 'N/A')}")
        
        # Create XML
        image_filename = txt_file.stem + '.jpg'
        objects = [data]
        xml_tree = create_pascal_voc_xml(image_filename, 1920, 1080, objects)
        
        # Save XML
        xml_filename = txt_file.stem + '.xml'
        xml_path = output_path / xml_filename
        
        try:
            xml_tree.write(str(xml_path), encoding='utf-8', xml_declaration=True)
            processed += 1
            print(f"  Created: {xml_filename}")
        except Exception as e:
            print(f"  Error saving: {e}")
            failed += 1
    
    print(f"\nResults: {processed} processed, {failed} failed")


if __name__ == "__main__":
    convert_mercosur_only()
