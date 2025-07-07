import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

import tqdm


def convert_to_obb(size, box):
    """
    Convert bounding box to OBB format (four corner points) normalized between 0-1
    For standard rectangular bounding box, we use the box corners in clockwise order:
    (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)
    
    Args:
        size: tuple (w, h) representing the image size
        box: tuple (xmin, xmax, ymin, ymax) representing the bounding box
    
    Returns:
        tuple: (x1, y1, x2, y2, x3, y3, x4, y4) representing the four corners normalized
    """
    image_width = size[0]
    image_height = size[1]
    
    # Extract bounding box coordinates
    xmin = box[0]
    xmax = box[1]
    ymin = box[2]
    ymax = box[3]
    
    # Make sure coordinates are valid
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    
    # Calculate the four corners of the bounding box
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    x1, y1 = xmin, ymin  # Top-left corner
    x2, y2 = xmax, ymin  # Top-right corner
    x3, y3 = xmax, ymax  # Bottom-right corner
    x4, y4 = xmin, ymax  # Bottom-left corner
    
    # Normalize coordinates to be between 0 and 1
    x1 = max(0, min(1, x1 / image_width))
    y1 = max(0, min(1, y1 / image_height))
    x2 = max(0, min(1, x2 / image_width))
    y2 = max(0, min(1, y2 / image_height))
    x3 = max(0, min(1, x3 / image_width))
    y3 = max(0, min(1, y3 / image_height))
    x4 = max(0, min(1, x4 / image_width))
    y4 = max(0, min(1, y4 / image_height))
    
    return (x1, y1, x2, y2, x3, y3, x4, y4)


def convert_annotation(input_file, output_file, classes):
    """
    Convert XML annotations to OBB format text files
    
    Args:
        input_file: path to XML annotation file
        output_file: path to output text file
        classes: dictionary mapping class names to class indices
    """
    try:
        in_file = open(input_file)
        out_file = open(output_file, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if difficult is not None:
                difficult = difficult.text
            
            # Try to find the class name using both 'name' and 'n' tags
            name_tag = obj.find('name')
            if name_tag is None:
                name_tag = obj.find('n')
                if name_tag is None:
                    # Silently skip objects without name tags
                    continue
                    
            cls = name_tag.text
            cls = cls.upper().replace(" ", "_")

            # Check if the class is in our classes dictionary
            if cls not in classes:
                # Silently skip classes not in our dictionary
                continue
                
            cls_id = classes[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            
            # Convert to OBB format (4 corner points)
            obb = convert_to_obb((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in obb]) + '\n')

        out_file.close()
        in_file.close()
    except Exception as ex:
        print(f"Error processing {input_file}: {str(ex)}")
        pass


def generate_labels_obb(dataset_path, classes, image_path='IMAGES'):
    """
    Generate OBB format labels for all XML annotations in the dataset
    
    Args:
        dataset_path: path to the main dataset directory
        classes: dictionary mapping class names to class indices
        image_path: relative path to the images subdirectory (default: 'IMAGES')
    """
    images = os.path.join(dataset_path, image_path)
    annotations_path = os.path.join(dataset_path, 'ANNOTATIONS')

    if not os.path.exists(annotations_path):
        print(f"Annotations directory not found: {annotations_path}")
        return

    annotations = os.listdir(annotations_path)

    print(f"Processing {len(annotations)} annotations in {dataset_path}...")
    for annotation in tqdm.tqdm(annotations):
        if 'xml' in annotation:
            annotation_xml = os.path.join(annotations_path, annotation)
            annotation_txt = os.path.join(images,
                                          annotation_xml.replace('xml', 'txt').split('/')[-1])
            convert_annotation(annotation_xml, annotation_txt, classes)
    print(f"Finished processing {dataset_path}")


if __name__ == '__main__':
    # Example usage
    generate_labels_obb('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/4.0',
                     classes={"MEAT": 0, "PERSON": 1, "SCRAP": 2, "HELMET": 3, "TRASH": 4, 'FULL_TRAY': 5, 'EMPTY_TRAY': 6})
    
    # Additional datasets can be added as needed:
    # generate_labels_obb('/path/to/dataset',
    #                   classes={"CLASS1": 0, "CLASS2": 1, ...})
