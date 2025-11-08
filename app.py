import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import subprocess

st.title("🐱 Cat Detector (Upload MP4)")
st.write("Upload an MP4 file and the app will detect cats, draw bounding boxes, and display the count on the video.")

# Upload video
uploaded_file = st.file_uploader("Upload your MP4 file", type=["mp4"])

if uploaded_file:
    # Save temporary video file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Upload completed!")

    # Load YOLO model
    model = YOLO("best.pt")

    # Read video
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    # Create folder to save frames
    frames_dir = tempfile.mkdtemp()
    frame_index = 0

    st.write("🐾 Detecting cats in the video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict
        results = model.predict(source=frame, conf=0.3, verbose=False)

        boxes = results[0].boxes
        class_indices = boxes.cls
        names = [model.names[int(cls)] for cls in class_indices]

        # Filter only cats
        cat_indices = [i for i, name in enumerate(names) if name == "cat"]
        cat_count = len(cat_indices)

        if cat_indices:
            annotated_frame = results[0].plot(boxes=boxes[cat_indices])  # Only cats
        else:
            annotated_frame = frame

        # Put text on frame
        text = f"Cats: {cat_count}"
        cv2.putText(annotated_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame{frame_index:05d}.png")
        cv2.imwrite(frame_path, annotated_frame)
        frame_index += 1

    cap.release()

    # Create output video via ffmpeg
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", os.path.join(frames_dir, "frame%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ])

    st.success("✅ Detection completed! Watch the result below")
    st.video(output_path)
