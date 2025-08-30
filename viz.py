import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

results = pd.read_csv('./test2.csv')
video_path = 'greenn.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out2.mp4', fourcc, fps, (width, height))

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            car_bbox_str = df_.iloc[row_indx]['car_bbox']
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str.strip())
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25, line_length_x=200, line_length_y=200)
            license_plate_bbox_str = df_.iloc[row_indx]['license_plate_bbox']
            x1, y1, x2, y2 = ast.literal_eval(license_plate_bbox_str.strip())
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
            plate_color = df_.iloc[row_indx]['license_color']
            plate_text = df_.iloc[row_indx]['license_number']
            color_text = f"{plate_text} ({plate_color})"
            (text_width, text_height), _ = cv2.getTextSize(color_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 6)
            text_x = int((car_x1 + car_x2 - text_width) / 2)
            text_y = int(y1) - 20 if int(y1) - 20 > 0 else 30
            cv2.putText(frame, color_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6)
        out.write(frame)
out.release()
cap.release()
cv2.destroyAllWindows()