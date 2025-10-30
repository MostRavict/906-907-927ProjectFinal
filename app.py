import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import subprocess

st.title("🐱 Cat Detector (Upload MP4)")
st.write("อัปโหลดไฟล์ MP4 ของคุณแล้วระบบจะตรวจจับแมวและวาด Bounding Boxes พร้อมจำนวนแมวบนวิดีโอ")

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

    # อ่าน video
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    # สร้างโฟลเดอร์เก็บ frames
    frames_dir = tempfile.mkdtemp()
    frame_index = 0
    frame_counts = []

    st.write("🐾 เริ่มตรวจจับแมวในวิดีโอ...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ตรวจทุก frame
        results = model.predict(source=frame, conf=0.3, verbose=False)

        boxes = results[0].boxes
        class_indices = boxes.cls
        names = [model.names[int(cls)] for cls in class_indices]

        # filter เฉพาะ cat
        cat_indices = [i for i, name in enumerate(names) if name == "cat"]
        cat_count = len(cat_indices)
        frame_counts.append(cat_count)

        if cat_indices:
            annotated_frame = results[0].plot(boxes=boxes[cat_indices])
        else:
            annotated_frame = frame

        # แปะข้อความจำนวนแมวบน frame
        text = f"แมว {cat_count} ตัว"
        cv2.putText(annotated_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # บันทึก frame เป็น PNG
        frame_path = os.path.join(frames_dir, f"frame{frame_index:05d}.png")
        cv2.imwrite(frame_path, annotated_frame)
        frame_index += 1

    cap.release()

    # สร้าง video ผ่าน ffmpeg
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", os.path.join(frames_dir, "frame%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ])

    st.success("✅ ตรวจจับเสร็จสิ้น! ดูวิดีโอผลลัพธ์ด้านล่าง")
    st.video(output_path)
