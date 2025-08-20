#!/usr/bin/env python3
"""
Test script to verify the OCR to Pascal VOC conversion works correctly.
"""

import tempfile
import os
from pathlib import Path
from convert_txt_to_pascal_voc import parse_annotation_file, create_pascal_voc_xml, process_annotations


def test_template_1():
    """Test parsing of template 1 format."""
    template_1_content = """camera: GoPro Hero4 Silver
position_vehicle: 750 316 314 275
	type: car
	make: Volvo
	model: XC60
	year: 2014
plate: AYU0035
corners: 872,436 949,437 950,462 871,462
	char 1: 876 445 8 14
	char 2: 885 445 9 14
	char 3: 895 445 8 13
	char 4: 908 445 9 14
	char 5: 918 446 8 13
	char 6: 927 446 8 13
	char 7: 936 446 9 13"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(template_1_content)
        temp_file = f.name
    
    try:
        # Parse the file
        data = parse_annotation_file(temp_file)
        
        print("Template 1 Test:")
        print(f"  Class name: {data['class_name']}")
        print(f"  Bounding box: {data['bbox']}")
        print(f"  Expected: brazilian_plate, (871, 436, 950, 462)")
        print(f"  Success: {data['class_name'] == 'brazilian_plate' and data['bbox'] == (871, 436, 950, 462)}")
        print()
        
    finally:
        os.unlink(temp_file)


def test_template_2():
    """Test parsing of template 2 format."""
    template_2_content = """type: car
plate: OVJ6688
layout: Brazilian
corners: 441,425 570,424 572,468 444,467"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(template_2_content)
        temp_file = f.name
    
    try:
        # Parse the file
        data = parse_annotation_file(temp_file)
        
        print("Template 2 Test (Brazilian):")
        print(f"  Class name: {data['class_name']}")
        print(f"  Bounding box: {data['bbox']}")
        print(f"  Expected: brazilian_plate, (441, 424, 572, 468)")
        print(f"  Success: {data['class_name'] == 'brazilian_plate' and data['bbox'] == (441, 424, 572, 468)}")
        print()
        
    finally:
        os.unlink(temp_file)


def test_template_2_mercosur():
    """Test parsing of template 2 format with Mercosur layout."""
    template_2_content = """type: car
plate: ABC1234
layout: Mercosur
corners: 100,100 200,100 200,150 100,150"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(template_2_content)
        temp_file = f.name
    
    try:
        # Parse the file
        data = parse_annotation_file(temp_file)
        
        print("Template 2 Test (Mercosur):")
        print(f"  Class name: {data['class_name']}")
        print(f"  Bounding box: {data['bbox']}")
        print(f"  Expected: mercosur_plate, (100, 100, 200, 150)")
        print(f"  Success: {data['class_name'] == 'mercosur_plate' and data['bbox'] == (100, 100, 200, 150)}")
        print()
        
    finally:
        os.unlink(temp_file)


def test_xml_generation():
    """Test XML generation."""
    objects = [{
        'class_name': 'brazilian_plate',
        'bbox': (441, 424, 572, 468)
    }]
    
    xml_tree = create_pascal_voc_xml('test_image.jpg', 1920, 1080, objects)
    
    # Create temporary XML file to verify structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        xml_tree.write(f.name, encoding='utf-8', xml_declaration=True)
        temp_xml = f.name
    
    try:
        with open(temp_xml, 'r') as f:
            xml_content = f.read()
        
        print("XML Generation Test:")
        print("  Generated XML structure:")
        print("  " + "\n  ".join(xml_content.split('\n')[:15]))  # Show first 15 lines
        print("  ...")
        print(f"  Contains brazilian_plate: {'brazilian_plate' in xml_content}")
        print(f"  Contains bbox coordinates: {'441' in xml_content and '572' in xml_content}")
        print()
        
    finally:
        os.unlink(temp_xml)


if __name__ == "__main__":
    print("Testing OCR to Pascal VOC Conversion")
    print("=" * 40)
    
    test_template_1()
    test_template_2()
    test_template_2_mercosur()
    test_xml_generation()
    
    print("All tests completed!")
