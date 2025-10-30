# app.py
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.title("🐱 Cat Detector (Upload MP4)")
st.write("อัปโหลดไฟล์ MP4 แล้วระบบจะตรวจจับแมวและวาด Bounding Boxes ให้ดู")

# Upload video
uploaded_file = st.file_uploader("อัปโหลดไฟล์ MP4 ของคุณ", type=["mp4"])

if uploaded_file:
    # บันทึกไฟล์ชั่วคราว
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ อัปโหลดเรียบร้อยแล้ว!")

    # โหลดโมเดล YOLO
    model = YOLO("yolo11n.pt")

    # สร้างไฟล์วิดีโอผลลัพธ์
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), fps, (width, height))

    st.write("🐾 เริ่มตรวจจับแมวในวิดีโอ...")

    frame_index = 0
    frame_interval = int(fps)  # ตรวจทุก 1 วินาที

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            results = model.predict(source=frame, conf=0.3, verbose=False)
            annotated_frame = results[0].plot()  # วาด Bounding Boxes
        else:
            annotated_frame = frame

        out.write(annotated_frame)
        frame_index += 1

    cap.release()
    out.release()

    st.success("✅ ตรวจจับเสร็จสิ้น! ดูวิดีโอผลลัพธ์ด้านล่าง")
    st.video(output_path)

