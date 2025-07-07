import cv2
import random
from ultralytics import YOLO

# Load YOLO model (use your custom model if needed)
model = YOLO("/home/diego/2TB/yolo/Trains/v8/object_detection/meat_and_person/GERAL_1.0/trains/nano/416/runs/detect/train_no_augmentatio_no_erase_and_no_crop_fraction_no_scale_no_translate_4_classes/weights/best.pt")

# Open video file
video_path = "/home/diego/Downloads/beter_beef_1.mp4"
cap = cv2.VideoCapture(video_path)

# Get class names from the model
class_names = model.names

# Generate unique colors for each class
random.seed(42)  # Ensures consistent colors on each run
colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(class_names))}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends

    # Perform left crop of 775 pixels
    cropped_frame = frame[:, 775:]  # Removes first 775 pixels from the left

    # Run YOLO object detection on cropped frame
    results = model(cropped_frame, conf=0.1, iou=0.1)

    # Iterate through all detected objects
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
            confidence = float(conf)  # Confidence score
            class_id = int(cls)  # Class index
            label = class_names[class_id]  # Class label
            color = colors[class_id]  # Get unique color for this class

            # Draw bounding box with class-specific color
            cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), color, 2)

            # Put label and confidence on the cropped frame
            text = f"{label} {confidence:.2f}"
            cv2.putText(cropped_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the cropped frame
    cv2.imshow("YOLOv8 Object Detection (Cropped)", cropped_frame)

    # Press 'q' to exit
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
