import argparse
import os
import cv2
from ultralytics import YOLO
import numpy as np


def get_class_colors(names):
    # Gera uma cor única para cada classe
    num_classes = len(names)
    np.random.seed(42)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]
    return colors

def draw_obb_vertices(frame, xyxyxyxy, conf, class_name=None, color=(0, 255, 0)):
    # xyxyxyxy: array-like, shape (8,)
    pts = np.array(xyxyxyxy, dtype=np.int32).reshape((-1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    label = f'{conf:.2f}'
    if class_name:
        label = f'{class_name}: {label}'
    pt_label = tuple(pts[0])
    cv2.putText(frame, label, (pt_label[0], pt_label[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_bbox(frame, xyxy, conf, class_name=None, color=(0, 255, 0)):
    # xyxy: array-like, shape (4,)
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f'{conf:.2f}'
    if class_name:
        label = f'{class_name}: {label}'
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_video(video_path, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    class_names = model.names if hasattr(model, 'names') else []
    class_colors = get_class_colors(class_names)
    frame_idx = 1
    debug_printed = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            # --- BOUNDING BOX CONVENCIONAL ---
            box_preds = getattr(result, 'boxes', None)
            if box_preds is not None and hasattr(box_preds, 'xyxy'):
                n_preds = len(box_preds.xyxy)
                if not debug_printed:
                    print(f"[DEBUG] {n_preds} bbox preds. xyxy shape: {getattr(box_preds.xyxy, 'shape', None)}")
                    debug_printed = True
                for i in range(n_preds):
                    xyxy = box_preds.xyxy[i].cpu().numpy() if hasattr(box_preds.xyxy[i], 'cpu') else box_preds.xyxy[i]
                    conf = float(box_preds.conf[i]) if hasattr(box_preds, 'conf') else 1.0
                    cls = int(box_preds.cls[i]) if hasattr(box_preds, 'cls') else None
                    class_name = class_names[cls] if cls is not None and cls < len(class_names) else str(cls)
                    color = class_colors[cls] if cls is not None and cls < len(class_colors) else (0, 255, 0)
                    draw_bbox(frame, xyxy, conf, class_name, color)
            else:
                if not debug_printed:
                    print(f"[DEBUG] result.boxes is None or missing xyxy!")
                    debug_printed = True
            # --- BOUNDING BOX ORIENTADO ---
            obb_preds = getattr(result, 'obb', None)
            if obb_preds is not None and hasattr(obb_preds, 'xyxyxyxy'):
                n_preds = len(obb_preds.xyxyxyxy)
                if not debug_printed:
                    print(f"[DEBUG] {n_preds} OBB preds. xyxyxyxy shape: {getattr(obb_preds.xyxyxyxy, 'shape', None)}")
                    debug_printed = True
                for i in range(n_preds):
                    xyxyxyxy = obb_preds.xyxyxyxy[i].cpu().numpy() if hasattr(obb_preds.xyxyxyxy[i], 'cpu') else obb_preds.xyxyxyxy[i]
                    conf = float(obb_preds.conf[i]) if hasattr(obb_preds, 'conf') else 1.0
                    cls = int(obb_preds.cls[i]) if hasattr(obb_preds, 'cls') else None
                    class_name = class_names[cls] if cls is not None and cls < len(class_names) else str(cls)
                    color = class_colors[cls] if cls is not None and cls < len(class_colors) else (0, 255, 0)
                    draw_obb_vertices(frame, xyxyxyxy, conf, class_name, color)
            else:
                if not debug_printed:
                    print(f"[DEBUG] result.obb is None or missing xyxyxyxy!")
                    debug_printed = True
        out_name = os.path.join(output_dir, f"{frame_idx:08d}.jpg")
        cv2.imwrite(out_name, frame)
        frame_idx += 1
    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Predict OBB model on video and save annotated frames.")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, required=True, help='Path to OBB Ultralytics model')
    parser.add_argument('--output', type=str, required=True, help='Directory to save output frames')
    args = parser.parse_args()
    process_video(args.video, args.model, args.output)

if __name__ == "__main__":
    main()
