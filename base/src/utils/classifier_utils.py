import cv2
from base.src.utils.detector_utils import DetectorUtils
from base.src.utils.image_classification_utils import ImageClassificationUtils

class ClassifierUtils:


    @staticmethod
    def predict(classifier, image, image_classification = False, get_intersection_score=False, threshold=0.05):

        image_height, image_width = image.shape[:2]
        prediction_results = classifier.predict(image, image_classification)
        if not image_classification:

            if get_intersection_score:
                best_result, intersection_score = DetectorUtils.get_best_detection(prediction_results, image_height,
                                                                                   image_width, get_intersection_score,
                                                                                   threshold)
                return best_result, intersection_score
            else:
                best_result = DetectorUtils.get_best_detection(prediction_results, image_height, image_width,
                                                               get_intersection_score, threshold)
                return best_result
            # best_result = ObjectDetectionUtils.get_best_result(prediction_results, image_height, image_width)
            # return best_result
        else:
            best_result = ImageClassificationUtils.get_best_result(prediction_results)
            return best_result


