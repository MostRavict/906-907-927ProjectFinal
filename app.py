import streamlit as st
import cv2
import numpy as np

st.title("📸 ทดสอบกล้องผ่าน browser")

img_file = st.camera_input("กดถ่ายภาพเพื่อทดสอบกล้อง")

if img_file is not None:
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.success("✅ กล้องทำงานได้!")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
