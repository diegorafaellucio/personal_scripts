from shapely.geometry import Polygon
import cv2
import os
import shutil
import json

def get_intersection_score(gt_coords, side_coords):
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_coords

    padding_polygon = Polygon(
        [(gt_x_min, gt_y_min), (gt_x_min, gt_y_max), (gt_x_max, gt_y_max), (gt_x_max, gt_y_min)])

    side_x_min, side_y_min, side_x_max, side_y_max = side_coords

    side_polygon = Polygon(
        [(side_x_min, side_y_min), (side_x_min, side_y_max), (side_x_max, side_y_max),
         (side_x_max, side_y_min)])

    intersection = padding_polygon.intersection(side_polygon)

    if intersection.area == 0:
        intersection_score = 0
    else:
        intersection_score = intersection.area / padding_polygon.area

    return intersection_score


def generate_json(label, image_name, json_path, counter, padding, image_path, classifier, not_annotated_path):

    try:
        img = cv2.imread(image_path)

        img_height, img_width = img.shape[:2]

        gt_x_min = padding
        gt_y_min = 0
        gt_x_max = img_width - padding
        gt_y_max = img_height

        gt_coords = [gt_x_min, gt_y_min, gt_x_max, gt_y_max]

        side_classification_results = classifier.detect(img)

        highest_intersection = 0
        highest_index = 0
        highest_coords = [0, 0, 0, 0]
        highest_label = ''

        for side_index, side_classification_result in enumerate(side_classification_results):
            # side_coords, side_classification_label, side_classification_confidence = side_classification_result

            x_min = side_classification_result['topleft']['x']
            y_min = side_classification_result['topleft']['y']

            x_max = side_classification_result['bottomright']['x']
            y_max = side_classification_result['bottomright']['y']

            label = side_classification_result['label'].split('-')[-1]

            side_coords = [x_min, y_min, x_max, y_max]

            # img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

            intersection_score = get_intersection_score(gt_coords, side_coords)

            if intersection_score > highest_intersection:
                highest_intersection = intersection_score
                highest_index = side_index
                highest_coords = side_coords
                highest_label = label

        if highest_intersection < 0.50:
            not_annotated_image_path = os.path.join(not_annotated_path, image_name)
            shutil.move(image_path, not_annotated_image_path)

        else:

            x_min, y_min, x_max, y_max = highest_coords



            template_json = {"version": "4.5.7",
                             "flags": {},
                             "shapes": [
                                {"label": label,
                                 "points": [[x_min, y_min],  [x_max, y_max]],
                                 "group_id": None,
                                    "shape_type": "rectangle",
                                 "flags": {}}],
                             "imagePath": '../IMAGES/' + image_name,
                             "imageData": None,
                             "imageHeight": img_height,
                             "imageWidth": img_width
                             }

            with open(json_path, "w") as outfile:
                json.dump(template_json, outfile, indent=4)

            return counter
    except Exception as ex:
        return counter

