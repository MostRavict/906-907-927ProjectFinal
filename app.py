import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.title("🐱 Real-time Cat Monitor")

# ให้ผู้ใช้ระบุจำนวนแมวที่คาดหวัง
expected_cats = st.number_input("จำนวนแมวที่คาดหวัง", min_value=1, step=1)

# โหลด YOLO model
model = YOLO("best.pt")

# ใช้กล้องผ่าน browser
st.write("📸 ถ่ายภาพแมวเพื่อทำการตรวจจับ")
img_file = st.camera_input("")

if img_file is not None:
    # แปลงไฟล์เป็นภาพ OpenCV
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # รัน YOLO ตรวจจับ
    results = model.predict(source=img, conf=0.3, verbose=False)
    boxes = results[0].boxes
    class_indices = boxes.cls
    names = [model.names[int(cls)] for cls in class_indices]

    # นับเฉพาะแมว
    cat_indices = [i for i, name in enumerate(names) if name == "cat"]
    cat_count = len(cat_indices)

    # วาดกรอบทั้งหมด
    annotated_frame = results[0].plot()

    # แสดงภาพ
    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # แสดงผลลัพธ์ใต้ภาพ
    if cat_count < expected_cats:
        st.error(f"🐾 พบแมว {cat_count} ตัว จากที่คาดไว้ {expected_cats} — Missing Cats!")
    else:
        st.success(f"🐾 พบแมวครบแล้ว {cat_count}/{expected_cats} ตัว — All Cats Present!")
