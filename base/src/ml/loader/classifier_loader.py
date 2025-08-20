from src.ml.classifier.pickle_classifier.classifier import Classifier as PickleDetector
from src.ml.classifier.yolo_classifier.classifier import Classifier as YoloClassifier
from src.ml.classifier.ultralytics_classifier.classifier import Classifier as UltralyticsClassifier
from src.enum.detection_approach_enum import DetectionApproachEnum
from django.conf import settings
import os


class ClassifierLoader():

    @staticmethod
    def load(model_path, detection_framework=DetectionApproachEnum.ULTRALYTICS.value):

        base_dir = settings.BASE_DIR

        detector = None

        if detection_framework == DetectionApproachEnum.ULTRALYTICS.value:
            model_path = os.path.join(base_dir, model_path )
            detector = UltralyticsClassifier(model_path)
        elif detection_framework == DetectionApproachEnum.YOLO.value:
            model_path = os.path.join(base_dir, model_path )
            detector = YoloClassifier(model_path)
        elif detection_framework == DetectionApproachEnum.PICKLE.value:
            model_path = os.path.join(base_dir, model_path)
            detector = PickleDetector(model_path)

        return detector





