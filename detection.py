import  cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3

engine=pyttsx3.init()
model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture(0)

vehicle_found_last=False

while True:
    ret,frame=cap.read()
    if not ret:
        print("--")
        break
    results=model(frame)
    vehicle_detected=False
    for result in results:
        for box in result.boxes:
            class_id=int(box.cls[0])
            name=model.names[class_id]
            if name in ["car","Truck","Bus","Motorcycle"]:
                vehicle_detected=True
                x1,y1,x2,y2=map(int,box.xyxy[0])

                cv2.rectangle(frame,
                              (x1,y1),
                              (x2,y2),
                              (0,255,0),
                              2)
                cv2.putText(frame,
                        name,
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

    if vehicle_detected:
        cv2.putText(frame,
                 "vehicle detected",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)
        if vehicle_found_last==False:
            engine.say("vehicle detected")
            engine.runAndWait()

        vehicle_found_last=True
    else:
        cv2.putText(frame,
                "no vehicle detected",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

        if vehicle_found_last==True:
            engine.say("no vehicle detected")
            engine.runAndWait()

        vehicle_found_last=False
        cv2.imshow("AI Vehicle detected",frame)
        if cv2.waitKey(1)& 0xFF==ord('q'):
             break
cap.release()
cv2.destroyAllWindows()