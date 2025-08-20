import logging
import dlib
import cv2
import numpy as np
from base.src.enum.cuts_enum import CutsEnum

class CutsUtils:
    logger = logging.getLogger(__name__)

    @staticmethod
    def get_cuts_with_classification_enabled():
        cuts_with_classification_enabled = []

        for item_key, item in CutsEnum.__members__.items():
            key = item.key
            model_name = item.model_name

            if model_name != "":
                cuts_with_classification_enabled.append(key)

        return cuts_with_classification_enabled



    @staticmethod
    def get_cuts(image, side_detection_result, side_a_shape_predictor, side_b_shape_predictor):
        # Check if side_detection_result is None or empty list
        if not side_detection_result:
            return {}  # Return empty dictionary if no detection

        side_detection_label = side_detection_result['label']

        cuts_coords = {}

        x_min = side_detection_result['topleft']['x']
        y_min = side_detection_result['topleft']['y']
        x_max = side_detection_result['bottomright']['x']
        y_max = side_detection_result['bottomright']['y']

        roi = image[y_min:y_max, x_min:x_max]

        rect = dlib.rectangle(left=0, top=0, right=roi.shape[1], bottom=roi.shape[0])

        if 'LADO_A' in side_detection_label:
            cuts_coords = side_a_shape_predictor.get_polygons(
                roi, rect, x_min, y_min)
        elif 'LADO_B' in side_detection_label:
            cuts_coords = side_b_shape_predictor.get_polygons(
                roi, rect, x_min, y_min)

        return cuts_coords

    @staticmethod
    def get_cuts_mask_and_cut_lines_image(cuts_coords, image):

        cut_lines_image = image.copy()
        cuts_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for cut_coord_key, cut_coord_data in cuts_coords.items():
            coords_polygon = cut_coord_data.reshape((-1, 1, 2))
            cv2.polylines(cut_lines_image, [coords_polygon], True, (255, 0, 255), 5)
            color = CutsEnum[cut_coord_key.upper()].value
            cv2.fillPoly(cuts_mask, pts=[coords_polygon], color=(color, color, color))

        _, binary_mask = cv2.threshold(cuts_mask, 0, 255, cv2.THRESH_BINARY)

        return cut_lines_image, cuts_mask, binary_mask

    @staticmethod
    def get_cut_image_without_background(cut_coords, image, cut_name):
        square_size = 512

        cut_coord_data = cut_coords[cut_name]

        p_min = np.min(cut_coord_data, axis=0)
        p_max = np.max(cut_coord_data, axis=0)

        p_min[p_min < 0] = 0
        p_max[p_max < 0] = 0

        points = np.reshape(cut_coord_data, (-1, 1, 2))
        mask = np.zeros((image.shape[0], image.shape[1]))
        mask = cv2.fillPoly(mask, [points], 255)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        mask_roi = mask[p_min[1]:p_max[1], p_min[0]:p_max[0]]

        mask_roi = cv2.resize(mask_roi, (square_size, square_size))
        _, mask_roi = cv2.threshold(mask_roi, 10, 255, cv2.THRESH_BINARY)
        mask_roi = np.uint8(mask_roi)

        source_roi = image[p_min[1]-1:p_max[1]+1, p_min[0]-1:p_max[0]+1]
        source_roi = cv2.resize(source_roi,(square_size,square_size))
        cut_image = cv2.bitwise_and(source_roi,source_roi, mask=mask_roi)


        return cut_image


    @staticmethod
    def get_cut_mask(image_shape, cut_coord_data):
        cuts_mask = np.zeros(image_shape,np.uint8)
        coords_polygon = cut_coord_data.reshape((-1, 1, 2))
        cv2.fillPoly(cuts_mask, pts=[coords_polygon], color=(255, 255, 255))
        _, binary_mask = cv2.threshold(cuts_mask, 0, 255, cv2.THRESH_BINARY)

        return binary_mask
