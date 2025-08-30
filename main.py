from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from util import read_license_plate, write_csv
import csv

def classify_number_plate(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([100, 255, 255])
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    green_pixels = cv2.countNonZero(green_mask)
    white_pixels = cv2.countNonZero(white_mask)
    return "Green" if green_pixels > white_pixels else "Not Green"

def write_csv_with_color(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score', 'license_color'])
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                car = results[frame_nmr][car_id]['car']
                plate = results[frame_nmr][car_id]['license_plate']
                writer.writerow([
                    frame_nmr,
                    car_id,
                    str(car['bbox']),
                    str(plate['bbox']),
                    plate['bbox_score'],
                    plate['text'],
                    plate['text_score'],
                    plate['color']
                ])

results = {}
tracker = DeepSort(max_age=50, n_init=2, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
cap = cv2.VideoCapture('./greenn.mp4')
vehicles = [2, 3, 5, 7]
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        detections = coco_model(frame)[0]
        detection_list = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detection_list.append(([x1, y1, x2, y2], score, int(class_id)))
        tracks = tracker.update_tracks(detection_list, frame=frame)
        track_ids = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            track_ids.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            for car in track_ids:
                xcar1, ycar1, xcar2, ycar2, car_id = car
                if xcar1 < x1 < x2 < xcar2 and ycar1 < y1 < y2 < ycar2:
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    if license_plate_text_score is None:
                        license_plate_text_score = 0.0
                    plate_color = classify_number_plate(license_plate_crop)
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [float(xcar1), float(ycar1), float(xcar2), float(ycar2)]},
                        'license_plate': {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'bbox_score': float(score),
                            'text': license_plate_text,
                            'text_score': float(license_plate_text_score),
                            'color': plate_color
                        }
                    }
                    break
results = {k: v for k, v in results.items() if v}
write_csv_with_color(results, './test2.csv')
