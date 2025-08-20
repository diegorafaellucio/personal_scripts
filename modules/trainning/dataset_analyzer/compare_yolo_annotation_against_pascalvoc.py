import os
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm


# Types
XYXY = Tuple[int, int, int, int]


def parse_pascal_voc(xml_path: str) -> Tuple[Tuple[int, int], List[Tuple[str, XYXY]]]:
    """Parse a Pascal VOC XML and return image size and list of (class_name, (xmin,ymin,xmax,ymax))."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_tag = root.find('size')
    if size_tag is None:
        raise ValueError(f"Missing <size> tag in {xml_path}")
    w = int(size_tag.find('width').text)
    h = int(size_tag.find('height').text)

    objects = []
    for obj in root.iter('object'):
        # class name can be under <name> or <n>
        name_tag = obj.find('name')
        if name_tag is None:
            name_tag = obj.find('n')
        if name_tag is None:
            # skip objects without name
            continue
        cls = name_tag.text.strip()

        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        # sanitize
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = max(xmin + 1, xmax), max(ymin + 1, ymax)

        objects.append((cls, (xmin, ymin, xmax, ymax)))

    return (w, h), objects


def yolo_to_xyxy(norm_x: float, norm_y: float, norm_w: float, norm_h: float, img_w: int, img_h: int) -> XYXY:
    """Convert YOLO normalized center format to absolute xyxy in pixels."""
    cx = norm_x * img_w
    cy = norm_y * img_h
    w = norm_w * img_w
    h = norm_h * img_h
    xmin = int(round(cx - w / 2.0))
    ymin = int(round(cy - h / 2.0))
    xmax = int(round(cx + w / 2.0))
    ymax = int(round(cy + h / 2.0))
    # clamp
    xmin = max(0, min(img_w - 1, xmin))
    ymin = max(0, min(img_h - 1, ymin))
    xmax = max(0, min(img_w - 1, xmax))
    ymax = max(0, min(img_h - 1, ymax))
    if xmax <= xmin:
        xmax = min(img_w - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(img_h - 1, ymin + 1)
    return xmin, ymin, xmax, ymax


def parse_yolo_txt(txt_path: str, img_w: int, img_h: int) -> List[Tuple[int, XYXY]]:
    """Parse YOLO txt file and return list of (cls_id, (xmin,ymin,xmax,ymax))."""
    results: List[Tuple[int, XYXY]] = []
    if not os.path.exists(txt_path):
        return results
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                # skip malformed
                continue
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            xyxy = yolo_to_xyxy(x, y, w, h, img_w, img_h)
            results.append((cls_id, xyxy))
    return results


def draw_boxes(
    img: np.ndarray,
    voc_boxes: List[Tuple[str, XYXY]],
    yolo_boxes: List[Tuple[int, XYXY]],
    id_to_name: Optional[Dict[int, str]] = None,
    color_voc: Tuple[int, int, int] = (0, 255, 0),
    color_yolo: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw VOC (green) and YOLO (red) boxes on a copy of the image."""
    vis = img.copy()
    # VOC boxes
    for cls_name, (xmin, ymin, xmax, ymax) in voc_boxes:
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color_voc, 2)
        label = f"VOC:{cls_name}"
        cv2.putText(vis, label, (xmin, max(0, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_voc, 1, cv2.LINE_AA)
    # YOLO boxes
    for cls_id, (xmin, ymin, xmax, ymax) in yolo_boxes:
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color_yolo, 2)
        name = id_to_name.get(cls_id, str(cls_id)) if id_to_name else str(cls_id)
        label = f"YOLO:{name}"
        cv2.putText(vis, label, (xmin, min(img.shape[0] - 1, ymax + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yolo, 1, cv2.LINE_AA)
    return vis


def find_image(images_dir: str, stem: str, exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')) -> Optional[str]:
    for ext in exts:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def parse_class_map(map_str: Optional[str]) -> Dict[int, str]:
    """Parse mapping like "BRAZILIAN_PLATE=0,MERCOSUR_PLATE=1" into {0: 'BRAZILIAN_PLATE', 1: 'MERCOSUR_PLATE'}."""
    if not map_str:
        return {0: 'BRAZILIAN_PLATE', 1: 'MERCOSUR_PLATE'}
    out: Dict[int, str] = {}
    for item in map_str.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            continue
        name, id_str = item.split('=', 1)
        try:
            out[int(id_str)] = name
        except ValueError:
            continue
    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO annotations against Pascal VOC XML by drawing both on images.')
    parser.add_argument('--dataset', type=str, default='/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0', help='Dataset root path')
    parser.add_argument('--images-subdir', type=str, default='IMAGES', help='Subdirectory for images and YOLO txts')
    parser.add_argument('--annotations-subdir', type=str, default='ANNOTATIONS', help='Subdirectory for Pascal VOC XMLs')
    parser.add_argument('--labels-subdir', type=str, default=None, help='Subdirectory for YOLO labels (defaults to images-subdir)')
    parser.add_argument('--output-dir', type=str, default='COMPARE_VOC_VS_YOLO', help='Where to save visual comparisons (under dataset root if relative)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files to process (0 = no limit)')
    parser.add_argument('--show', action='store_true', help='Show each comparison and wait for a key press (cv2.waitKey(0))')
    parser.add_argument('--class-map', type=str, default='BRAZILIAN_PLATE=0,MERCOSUR_PLATE=1', help='Mapping like NAME=ID,NAME=ID to decode YOLO ids')
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset, args.images_subdir)
    annotations_dir = os.path.join(args.dataset, args.annotations_subdir)
    labels_dir = os.path.join(args.dataset, args.labels_subdir or args.images_subdir)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(args.dataset, output_dir)
    ensure_dir(output_dir)

    id_to_name = parse_class_map(args.class_map)

    xml_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.xml')]
    processed = 0
    skipped_missing = 0
    mismatches = 0

    for xml_file in tqdm(xml_files, desc='Comparing'):
        stem = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(annotations_dir, xml_file)

        try:
            (img_w, img_h), voc_boxes = parse_pascal_voc(xml_path)
        except Exception as e:
            print(f"[WARN] Skipping {xml_path}: {e}")
            skipped_missing += 1
            continue

        img_path = find_image(images_dir, stem)
        if img_path is None:
            print(f"[WARN] Image not found for {stem} in {images_dir}")
            skipped_missing += 1
            continue

        # YOLO txt is expected in labels_dir with same stem
        txt_path = os.path.join(labels_dir, stem + '.txt')
        if not os.path.exists(txt_path):
            print(f"[WARN] YOLO txt not found: {txt_path}")
            skipped_missing += 1
            continue

        # Read image to draw
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            skipped_missing += 1
            continue
        # Sanity check on dimension agreement
        if img.shape[1] != img_w or img.shape[0] != img_h:
            # not necessarily an error; just draw using actual image size for both
            img_w, img_h = img.shape[1], img.shape[0]

        yolo_boxes = parse_yolo_txt(txt_path, img_w, img_h)
        if not yolo_boxes and not voc_boxes:
            skipped_missing += 1
            continue

        vis = draw_boxes(img, voc_boxes, yolo_boxes, id_to_name=id_to_name)

        # No resizing: keep original image size for visualization

        out_path = os.path.join(output_dir, stem + '.jpg')
        cv2.imwrite(out_path, vis)
        processed += 1

        if args.show:
            cv2.imshow('VOC vs YOLO', vis)
            # Wait indefinitely for a key press to step through comparisons
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC to stop
                break

        if args.limit and processed >= args.limit:
            break

    if args.show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"Done. Processed: {processed}, Skipped: {skipped_missing}, Output: {output_dir}")


if __name__ == '__main__':
    main()

