import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

th1 = st.slider("Threshold1", 0, 1000, 100)
th2 = st.slider("Threshold2", 0, 1000, 200)


def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")

    image = cv2.cvtColor(cv2.Canny(image, th1, th2), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)
