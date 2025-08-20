#!/usr/bin/env python3
"""
Script to analyze Mercosur plate dimensions and compare with Brazilian plates.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from convert_txt_to_pascal_voc import parse_annotation_file


def analyze_plate_dimensions():
    """Analyze dimensions of Brazilian vs Mercosur plates."""
    
    input_dir = "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_TXT/"
    input_path = Path(input_dir)
    
    brazilian_plates = []
    mercosur_plates = []
    
    print("Analyzing plate dimensions...")
    
    # Sample some files for analysis
    count = 0
    for txt_file in input_path.glob('*.txt'):
        if count >= 100:  # Analyze first 100 files
            break
            
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
            
            data = parse_annotation_file(str(txt_file))
            if not data:
                continue
                
            bbox = data['bbox']  # (xmin, ymin, xmax, ymax)
            width = bbox[2] - bbox[0]   # xmax - xmin
            height = bbox[3] - bbox[1]  # ymax - ymin
            aspect_ratio = width / height if height > 0 else 0
            
            plate_info = {
                'file': txt_file.name,
                'class': data['class_name'],
                'bbox': bbox,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'coordinates': data.get('coordinates', [])
            }
            
            if data['class_name'] == 'brazilian_plate':
                brazilian_plates.append(plate_info)
            elif data['class_name'] == 'mercosur_plate':
                mercosur_plates.append(plate_info)
                
            count += 1
            
        except Exception as e:
            continue
    
    print(f"\nAnalyzed {len(brazilian_plates)} Brazilian plates and {len(mercosur_plates)} Mercosur plates")
    
    # Analyze Brazilian plates
    if brazilian_plates:
        print(f"\n{'='*50}")
        print("BRAZILIAN PLATES ANALYSIS")
        print(f"{'='*50}")
        
        avg_width = sum(p['width'] for p in brazilian_plates) / len(brazilian_plates)
        avg_height = sum(p['height'] for p in brazilian_plates) / len(brazilian_plates)
        avg_aspect = sum(p['aspect_ratio'] for p in brazilian_plates) / len(brazilian_plates)
        
        print(f"Average dimensions: {avg_width:.1f} x {avg_height:.1f}")
        print(f"Average aspect ratio: {avg_aspect:.2f}")
        
        print("\nSample Brazilian plates:")
        for i, plate in enumerate(brazilian_plates[:5]):
            print(f"  {plate['file']}: {plate['width']} x {plate['height']} (ratio: {plate['aspect_ratio']:.2f})")
            print(f"    Coordinates: {plate['coordinates']}")
            print(f"    Bbox: {plate['bbox']}")
    
    # Analyze Mercosur plates
    if mercosur_plates:
        print(f"\n{'='*50}")
        print("MERCOSUR PLATES ANALYSIS")
        print(f"{'='*50}")
        
        avg_width = sum(p['width'] for p in mercosur_plates) / len(mercosur_plates)
        avg_height = sum(p['height'] for p in mercosur_plates) / len(mercosur_plates)
        avg_aspect = sum(p['aspect_ratio'] for p in mercosur_plates) / len(mercosur_plates)
        
        print(f"Average dimensions: {avg_width:.1f} x {avg_height:.1f}")
        print(f"Average aspect ratio: {avg_aspect:.2f}")
        
        print("\nSample Mercosur plates:")
        for i, plate in enumerate(mercosur_plates[:5]):
            print(f"  {plate['file']}: {plate['width']} x {plate['height']} (ratio: {plate['aspect_ratio']:.2f})")
            print(f"    Coordinates: {plate['coordinates']}")
            print(f"    Bbox: {plate['bbox']}")
    
    # Compare aspect ratios
    if brazilian_plates and mercosur_plates:
        print(f"\n{'='*50}")
        print("COMPARISON")
        print(f"{'='*50}")
        
        br_avg_aspect = sum(p['aspect_ratio'] for p in brazilian_plates) / len(brazilian_plates)
        me_avg_aspect = sum(p['aspect_ratio'] for p in mercosur_plates) / len(mercosur_plates)
        
        print(f"Brazilian average aspect ratio: {br_avg_aspect:.2f}")
        print(f"Mercosur average aspect ratio: {me_avg_aspect:.2f}")
        
        if abs(br_avg_aspect - me_avg_aspect) > 0.5:
            print("⚠️  SIGNIFICANT DIFFERENCE IN ASPECT RATIOS DETECTED!")
            if br_avg_aspect > me_avg_aspect:
                print("   Brazilian plates are wider relative to height")
            else:
                print("   Mercosur plates are wider relative to height")
        else:
            print("✅ Aspect ratios are similar")


def check_coordinate_order():
    """Check if coordinates are in different order for Mercosur vs Brazilian."""
    
    input_dir = "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/ANNOTATIONS_TXT/"
    
    # Check a few specific files
    test_files = [
        "cars-me_img_010011.txt",  # Mercosur
        "cars-me_img_010028.txt",  # Mercosur
    ]
    
    print(f"\n{'='*50}")
    print("COORDINATE ORDER ANALYSIS")
    print(f"{'='*50}")
    
    for filename in test_files:
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        print(f"\nFile: {filename}")
        print("Content:")
        print(content)
        
        data = parse_annotation_file(filepath)
        if data:
            coords = data['coordinates']
            bbox = data['bbox']
            
            print(f"Original coordinates: {coords}")
            print(f"Calculated bbox: {bbox}")
            
            # Check if coordinates form a rectangle
            if len(coords) == 4:
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                
                print(f"X range: {min(x_coords)} to {max(x_coords)} (width: {max(x_coords) - min(x_coords)})")
                print(f"Y range: {min(y_coords)} to {max(y_coords)} (height: {max(y_coords) - min(y_coords)})")


if __name__ == "__main__":
    analyze_plate_dimensions()
    check_coordinate_order()
