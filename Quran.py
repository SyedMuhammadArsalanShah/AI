import streamlit as st
import cv2
import numpy as np
import requests
import time

# ------------------- UI CONFIG -------------------
st.set_page_config(page_title="🕌 Quran Gesture Player (OpenCV Only)", layout="wide")
st.title("🕌 Quran Gesture Player – Hand Controlled (No MediaPipe)")
st.markdown("""
### 🎮 Controls:
- 👍 **Thumb Up** → ▶️ Play Surah  
- 👎 **Thumb Down** → ⏸️ Pause  
- ✋ **Swipe Right** → ⏭️ Next Surah  
- ✋ **Swipe Left** → ⏮️ Previous Surah  
---
""")

# ------------------- FETCH SURAH LIST -------------------
list_url = "https://api.alquran.cloud/v1/surah"
res = requests.get(list_url).json()

if res["status"] == "OK":
    surah_list = res["data"]
    surah_options = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in surah_list]
else:
    st.error("❌ Failed to fetch Surah list.")
    st.stop()

selected_surah = st.selectbox("📖 Select a Surah", surah_options)
current_surah = int(selected_surah.split(".")[0])

# ------------------- FETCH SELECTED SURAH -------------------
def fetch_surah(num):
    url = f"https://api.alquran.cloud/v1/surah/{num}/ar.alafasy"
    r = requests.get(url).json()
    if r["status"] == "OK":
        return r["data"]
    else:
        return None

surah_data = fetch_surah(current_surah)
if not surah_data:
    st.error("Could not fetch Surah data.")
    st.stop()

status_box = st.empty()
audio_box = st.empty()
FRAME_WINDOW = st.image([])

# ------------------- GESTURE DETECTION -------------------
def detect_gesture(frame, prev_center):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, prev_center

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 6000:
        return None, prev_center

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / float(w)
    center = (x + w//2, y + h//2)

    gesture = None

    # Thumb Up / Down
    if aspect_ratio > 1.5:
        gesture = "up"
    elif aspect_ratio < 0.7:
        gesture = "down"

    # Swipe
    if prev_center is not None:
        dx = center[0] - prev_center[0]
        if dx > 100:
            gesture = "right"
        elif dx < -100:
            gesture = "left"

    return gesture, center

# ------------------- CAMERA CONTROL -------------------
run = st.checkbox("🎥 Start Camera Control")
camera = None
is_playing = False
prev_center = None
last_gesture_time = time.time()

if run:
    camera = cv2.VideoCapture(0)
    st.info("📷 Use your hand to control playback!")

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("⚠️ Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    gesture, prev_center = detect_gesture(frame, prev_center)
    now = time.time()

    if gesture and (now - last_gesture_time > 1.2):
        last_gesture_time = now

        if gesture == "up":
            is_playing = True
            status_box.success(f"▶️ Playing {surah_data['englishName']}")
            audio_box.audio(surah_data["ayahs"][0]["audio"])

        elif gesture == "down":
            is_playing = False
            status_box.warning("⏸️ Paused Surah")
            audio_box.empty()

        elif gesture == "right":
            current_surah = min(current_surah + 1, 114)
            surah_data = fetch_surah(current_surah)
            st.subheader(f"📖 {surah_data['englishName']} ({surah_data['name']})")
            status_box.info("⏭️ Next Surah")

        elif gesture == "left":
            current_surah = max(current_surah - 1, 1)
            surah_data = fetch_surah(current_surah)
            st.subheader(f"📖 {surah_data['englishName']} ({surah_data['name']})")
            status_box.info("⏮️ Previous Surah")

    cv2.putText(frame, f"Gesture: {gesture or 'None'}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    FRAME_WINDOW.image(frame, channels="BGR")

if camera:
    camera.release()
