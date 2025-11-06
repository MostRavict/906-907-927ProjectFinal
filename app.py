import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np

st.title("🐱 Real-time Cat Monitor")

expected_cats = st.number_input("จำนวนแมวที่คาดหวัง", min_value=1, step=1)

# โหลด YOLO model
model = YOLO("yolo11n.pt")

# ปุ่มเลือกโหมด
mode = st.radio("เลือกรูปแบบการตรวจจับ", ["🎥 กล้อง (Local Only)", "📸 กล้องผ่าน Browser (ใช้ได้ทุกที่)"])

# ==========
# โหมด 1: ใช้ webcam (เฉพาะ local)
# ==========
if mode == "🎥 กล้อง (Local Only)":
    run = st.checkbox("▶️ เริ่มตรวจจับผ่านกล้อง local")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning("⚠️ ไม่สามารถเปิดกล้องได้ (ตรวจสอบสิทธิ์หรือ index)")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ ไม่สามารถอ่านภาพจากกล้องได้")
                break

            results = model.predict(source=frame, conf=0.3, verbose=False)
            boxes = results[0].boxes
            class_indices = boxes.cls
            names = [model.names[int(cls)] for cls in class_indices]

            cat_indices = [i for i, name in enumerate(names) if name == "cat"]
            cat_count = len(cat_indices)

            annotated_frame = results[0].plot()

            text = f"Cats: {cat_count}"
            if cat_count < expected_cats:
                color = (0, 0, 255)
                status_text = "🐾 Missing Cats!"
            else:
                color = (0, 255, 0)
                status_text = "🐾 All Cats Present!"

            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(annotated_frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.1)

        cap.release()

# ==========
# โหมด 2: ใช้กล้อง browser (ใช้ได้ทั้ง local/cloud)
# ==========
else:
    img_file = st.camera_input("📸 ถ่ายภาพเพื่อตรวจจับแมว")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(source=img, conf=0.3, verbose=False)
        boxes = results[0].boxes
        class_indices = boxes.cls
        names = [model.names[int(cls)] for cls in class_indices]

        cat_indices = [i for i, name in enumerate(names) if name == "cat"]
        cat_count = len(cat_indices)

        annotated_frame = results[0].plot()
        text = f"Cats: {cat_count}"
        if cat_count < expected_cats:
            color = (0, 0, 255)
            status_text = "🐾 Missing Cats!"
        else:
            color = (0, 255, 0)
            status_text = "🐾 All Cats Present!"

        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated_frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
