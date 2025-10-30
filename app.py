import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import numpy as np

st.title("🐱 Real-time Cat Monitor")
expected_cats = st.number_input("จำนวนแมวที่คาดหวัง", min_value=1, step=1)

# โหลด YOLO
model = YOLO("yolo11n.pt")

# ใช้ webcam
stframe = st.empty()
cap = cv2.VideoCapture(0)  # 0 = webcam

st.write("เริ่มตรวจจับแมว... กด Stop เพื่อหยุด")

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("ไม่สามารถเข้ากล้องได้")
        break

    results = model.predict(source=frame, conf=0.3, verbose=False)
    boxes = results[0].boxes
    class_indices = boxes.cls
    names = [model.names[int(cls)] for cls in class_indices]

    # filter เฉพาะ cat
    cat_indices = [i for i, name in enumerate(names) if name == "cat"]
    cat_count = len(cat_indices)

    # วาดเฉพาะแมว
    if cat_indices:
        cat_boxes = boxes[cat_indices]
        annotated_frame = results[0].plot(boxes=cat_boxes)
    else:
        annotated_frame = frame

    # แสดงข้อความจำนวนแมว
    text = f"Cats: {cat_count}"
    if cat_count < expected_cats:
        color = (0, 0, 255)  # แดง
        status_text = "🐾 Missing Cats!"
    else:
        color = (0, 255, 0)  # เขียว
        status_text = "🐾 All Cats Present!"

    cv2.putText(annotated_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(annotated_frame, status_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # เพิ่ม delay เล็กน้อย
    time.sleep(0.1)
