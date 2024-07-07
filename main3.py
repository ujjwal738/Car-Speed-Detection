import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time
from math import dist
import matplotlib.pyplot as plt

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('veh2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()

offset = 6

cy1 = 322
cy2 = 368

veh_down = {}
counter = []
veh_down_time = {}

veh_up = {}
counter1 = []
veh_up_time = {}

speed_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            veh_down[id] = time.time()
        if id in veh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - veh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    speed_data.append((id, a_speed_kh))
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255, 255, 255), 2)


        if cy2 < (cy + offset) and cy2 > (cy - offset):
            veh_up[id] = time.time()
        if id in veh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed_time_1 = time.time() - veh_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10
                    a_speed_ms1 = distance1 / elapsed_time_1
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    speed_data.append((id, a_speed_kh1))
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255, 255, 255), 2)
    
    cv2.line(frame, (274, cy1), (814, cy1), (0, 255, 255), 1)
    cv2.putText(frame, ('LINE 1'), (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (34, 139, 34), 2)
    cv2.line(frame, (177, cy2), (927, cy2), (0, 255, 255), 1)
    cv2.putText(frame, ('LINE 2'), (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (34, 139, 34), 2)

    length_down = (len(counter))
    cv2.putText(frame, ('Car Down:') + str(length_down), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255),2)

    length_up = (len(counter1))
    cv2.putText(frame, ('Car Up:') + str(length_up), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

car_ids = [data[0] for data in speed_data]
speeds = [data[1] for data in speed_data]

plt.figure(figsize=(10, 6))
plt.plot(car_ids, speeds, 'bo-')
plt.xlabel('Car ID')
plt.ylabel('Speed (Km/h)')
plt.title('Car Speeds')
plt.grid(True)
plt.show()