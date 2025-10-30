import streamlit as st
import yt_dlp
import cv2
from ultralytics import YOLO
import tempfile
import os

st.title("🐱 YouTube Cat Detector")
st.write("ใส่ลิงก์ YouTube แล้วระบบจะตรวจจับ 'แมว' และแสดง Bounding Boxes ให้ดู!")

# Input YouTube URL
url = st.text_input("🔗 วางลิงก์ YouTube ที่นี่")

if url:
    with st.spinner("กำลังดาวน์โหลดวิดีโอ..."):
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        ydl_opts = {"format": "best[ext=mp4]", "outtmpl": temp_video.name, "quiet": True}
        yt_dlp.YoutubeDL(ydl_opts).download([url])

    st.success("✅ ดาวน์โหลดวิดีโอเรียบร้อย!")

    # โหลดโมเดล YOLO
    model = YOLO("yolov8n.pt")

    # สร้างวิดีโอผลลัพธ์ชั่วคราว
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cap = cv2.VideoCapture(temp_video.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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

    st.success("✅ ตรวจจับเสร็จสิ้น! นี่คือวิดีโอที่มี Bounding Boxes ของแมว 🐈")
    st.video(output_path)
