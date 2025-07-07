import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil


def convert(size, box):
    """
    Convert VOC annotation format to YOLO format
    
    Args:
        size: (width, height) of the image
        box: (xmin, xmax, ymin, ymax)
        
    Returns:
        (x, y, w, h) normalized coordinates for YOLO
    """
    image_width = size[0]
    image_height = size[1]
    
    # Check for invalid values
    if image_width <= 0 or image_height <= 0:
        print(f"WARNING: Invalid image dimensions: {size}")
        return None
    
    # Make sure box coordinates are in correct order
    xmin = min(box[0], box[1])
    xmax = max(box[0], box[1])
    ymin = min(box[2], box[3])
    ymax = max(box[2], box[3])
    
    # Ensure bounding box is valid
    if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
        print(f"WARNING: Invalid bounding box coordinates: {box}, image size: {size}")
        # Clip coordinates to image boundaries
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image_width, xmax)
        ymax = min(image_height, ymax)
    
    # Check if box has area
    if xmax <= xmin or ymax <= ymin:
        print(f"WARNING: Zero area bounding box: {box}")
        return None

    # Calculate YOLO format (x_center, y_center, width, height) - all normalized
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    # Normalize to 0-1
    x = x / image_width
    y = y / image_height
    w = w / image_width
    h = h / image_height

    # Double check normalization is correct (should be 0-1)
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        print(f"WARNING: Normalized coordinates out of range: ({x}, {y}, {w}, {h})")
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

    return (x, y, w, h)


def convert_annotation(input_file, output_file, subset, classes):
    """
    Convert XML annotation to YOLO txt format
    
    Args:
        input_file: Path to XML file
        output_file: Path to output txt file
        subset: Class subset/folder name
        classes: Dictionary mapping class names to class IDs
    """
    try:
        in_file = open(input_file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            print(f"ERROR: No size element in {input_file}")
            return False
            
        try:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if w <= 0 or h <= 0:
                print(f"ERROR: Invalid image dimensions in {input_file}")
                return False
        except (ValueError, AttributeError) as e:
            print(f"ERROR: Could not parse image size in {input_file}: {e}")
            return False
        
        # Only open output file if we have valid objects to write
        objects_to_write = []
        
        for obj in root.iter('object'):
            try:
                cls = obj.find('name').text.upper() if obj.find('name') is not None else None
                
                # If we're using the folder name as the class, then use that instead
                if subset in classes:
                    cls_id = classes[subset]
                elif cls in classes:
                    cls_id = classes[cls]
                else:
                    print(f"WARNING: Class '{cls}' not found in classes dictionary for file {input_file}")
                    continue
                    
                xmlbox = obj.find('bndbox')
                if xmlbox is None:
                    print(f"WARNING: No bndbox element for object in {input_file}")
                    continue
                    
                try:
                    xmin = float(xmlbox.find('xmin').text)
                    xmax = float(xmlbox.find('xmax').text)
                    ymin = float(xmlbox.find('ymin').text)
                    ymax = float(xmlbox.find('ymax').text)
                except (ValueError, AttributeError) as e:
                    print(f"ERROR: Invalid bounding box coordinates in {input_file}: {e}")
                    continue
                
                b = (xmin, xmax, ymin, ymax)
                bb = convert((w, h), b)
                
                if bb is not None:
                    objects_to_write.append((cls_id, bb))
            except Exception as e:
                print(f"ERROR processing object in {input_file}: {e}")
                continue
        
        # Only write to file if we have valid objects
        if objects_to_write:
            with open(output_file, 'w') as out_file:
                for cls_id, bb in objects_to_write:
                    out_file.write(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")
            return True
        else:
            print(f"No valid objects found in {input_file}")
            # Create an empty file to indicate it was processed but had no valid objects
            with open(output_file, 'w') as out_file:
                pass
            return False
            
    except Exception as e:
        print(f"ERROR processing file {input_file}: {e}")
        return False


def generate_labels(dataset_path, classes, image_path='IMAGES'):
    """
    Generate YOLO labels for all XML annotations in dataset
    
    Args:
        dataset_path: Path to dataset root
        classes: Dictionary mapping class names to class IDs
        image_path: Name of the images directory
    """
    valid_count = 0
    invalid_count = 0
    processed_count = 0
    
    print(f"Generating labels for {dataset_path}")
    print(f"Classes: {classes}")
    
    for dataset, persons, _ in os.walk(dataset_path):
        for subset in persons:
            # Check if this subset is in our classes dictionary
            if subset in classes:
                person_path = os.path.join(dataset, subset)
                images_dir = os.path.join(person_path, image_path)
                annotations_path = os.path.join(person_path, 'ANNOTATIONS')
                
                # Verify directories exist
                if not os.path.exists(images_dir):
                    print(f"ERROR: Images directory not found: {images_dir}")
                    continue
                    
                if not os.path.exists(annotations_path):
                    print(f"ERROR: Annotations directory not found: {annotations_path}")
                    continue
                
                print(f"Processing {subset} annotations...")
                
                # Get list of XML files
                try:
                    annotations = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]
                except Exception as e:
                    print(f"ERROR listing annotations: {e}")
                    continue
                    
                for annotation in annotations:
                    processed_count += 1
                    annotation_xml = os.path.join(annotations_path, annotation)
                    annotation_txt = os.path.join(images_dir, annotation.replace('.xml', '.txt'))
                    
                    if os.path.exists(annotation_xml):
                        if convert_annotation(annotation_xml, annotation_txt, subset, classes):
                            valid_count += 1
                        else:
                            invalid_count += 1
                    else:
                        print(f"XML file not found: {annotation_xml}")
                        invalid_count += 1
        
        # Only process the first level of directories
        break
    
    print(f"Annotation processing complete.")
    print(f"Total processed: {processed_count}, Valid: {valid_count}, Invalid: {invalid_count}")


if __name__ == '__main__':
    # You can try running with one class at a time to identify issues
    # Make sure all referenced classes exist in your dataset and are correctly named
    generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/BARRA_MANSA/BM/4.0',
                    image_path='IMAGES', classes={"AUSENTE": 0, "ESCASSA": 1, "MEDIANA": 2, "UNIFORME": 3, "EXCESSIVA": 4})
    
    # Uncomment to verify that annotations are valid for another dataset
    generate_labels('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0',
                   image_path='IMAGES', classes={"AUSENTE": 0, "ESCASSA": 1, "MEDIANA": 2, "UNIFORME": 3, "EXCESSIVA": 4})
