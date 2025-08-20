#!/usr/bin/env python3
"""
Debug script to investigate Mercosur coordinate conversion issues.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from convert_txt_to_pascal_voc import parse_annotation_file, create_pascal_voc_xml
from xml.etree.ElementTree import tostring


def debug_mercosur_files():
    """Debug specific Mercosur annotation files."""
    
    # Test files found with Mercosur layout
    test_files = [
        "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_TXT/cars-me_img_010011.txt",
        "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_TXT/cars-me_img_010028.txt",
        "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_TXT/cars-me_img_010002.txt"
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Debugging file: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # Read original content
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        print("Original content:")
        print(original_content)
        print("-" * 40)
        
        # Parse the file
        data = parse_annotation_file(file_path)
        
        if data:
            print("Parsed data:")
            print(f"  Class name: {data['class_name']}")
            print(f"  Original coordinates: {data.get('coordinates', 'N/A')}")
            print(f"  Bounding box (xmin, ymin, xmax, ymax): {data['bbox']}")
            
            # Manual coordinate calculation for verification
            if 'coordinates' in data:
                coords = data['coordinates']
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                
                print(f"  Manual verification:")
                print(f"    X coordinates: {x_coords}")
                print(f"    Y coordinates: {y_coords}")
                print(f"    Min X: {min(x_coords)}, Max X: {max(x_coords)}")
                print(f"    Min Y: {min(y_coords)}, Max Y: {max(y_coords)}")
                print(f"    Expected bbox: ({min(x_coords)}, {min(y_coords)}, {max(x_coords)}, {max(y_coords)})")
            
            # Generate XML
            objects = [data]
            xml_tree = create_pascal_voc_xml(f"{os.path.basename(file_path).replace('.txt', '.jpg')}", 
                                           1920, 1080, objects)
            
            # Show XML content
            xml_content = tostring(xml_tree.getroot(), encoding='unicode')
            print(f"\nGenerated XML:")
            print(xml_content)
            
        else:
            print("Failed to parse file!")


def test_coordinate_parsing():
    """Test coordinate parsing with known values."""
    
    print("\n" + "="*60)
    print("TESTING COORDINATE PARSING")
    print("="*60)
    
    # Test case from the Mercosur file
    test_coords = "520,358 640,357 642,396 521,397"
    
    from convert_txt_to_pascal_voc import parse_coordinates, get_bbox_from_coords
    
    print(f"Input coordinates: {test_coords}")
    
    coords = parse_coordinates(test_coords)
    print(f"Parsed coordinates: {coords}")
    
    bbox = get_bbox_from_coords(coords)
    print(f"Calculated bbox: {bbox}")
    
    # Manual verification
    if coords:
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        expected_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        print(f"Expected bbox: {expected_bbox}")
        print(f"Match: {bbox == expected_bbox}")


if __name__ == "__main__":
    print("DEBUGGING MERCOSUR COORDINATE CONVERSION")
    print("="*60)
    
    test_coordinate_parsing()
    debug_mercosur_files()
