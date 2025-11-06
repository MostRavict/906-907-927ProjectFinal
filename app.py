import streamlit as st
import cv2
from ultralytics import YOLO
import time

st.title("🐱 Real-time Cat Monitor")

model = YOLO("yolo11n.pt")

run = st.checkbox("▶️ เริ่มกล้อง")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(1)
print("Camera opened:", cap.isOpened())
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("ไม่สามารถเข้ากล้องได้")
        break

    results = model.predict(source=frame, conf=0.3, verbose=False)
    annotated = results[0].plot()

    FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    time.sleep(0.1)

cap.release()
