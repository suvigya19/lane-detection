import streamlit as st
import os
import cv2
import numpy as np
import shutil
from natsort import natsorted
import tempfile
import time

def process_video(input_video_path, frame_dir, output_dir, output_video_path):
    """Process the video and apply lane detection."""
    # Extract frames from input video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Use input video's FPS
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{i}.png"), frame)
        i += 1
    cap.release()

    # Get and sort frames
    frames = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
    frames = natsorted(frames)
    if not frames:
        st.error("No frames extracted from the video. Please check the video file.")
        return False

    # Create polygon mask from first frame
    sample_img = cv2.imread(os.path.join(frame_dir, frames[0]))
    h, w = sample_img.shape[:2]
    stencil = np.zeros((h, w), dtype=np.uint8)
    polygon = np.array([[50, 270], [220, 160], [360, 160], [480, 270]])
    cv2.fillConvexPoly(stencil, polygon, 1)

    # Lane detection with progress bar
    progress_bar = st.progress(0)
    for idx, fname in enumerate(frames):
        img = cv2.imread(os.path.join(frame_dir, fname))
        masked = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=stencil)
        _, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=200)
        out_img = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(out_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imwrite(os.path.join(output_dir, f"{idx}.png"), out_img)
        progress_bar.progress((idx + 1) / len(frames))

    # Combine frames into output video
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    output_files = natsorted(output_files)
    frame_list = [cv2.imread(os.path.join(output_dir, f)) for f in output_files]
    height, width, _ = frame_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in frame_list:
        out.write(frame)
    out.release()
    return True

def main():
    st.title("Lane Detection App")
    st.write("Upload a video to detect lanes and get the processed output.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Create unique directories using timestamp
        timestamp = str(int(time.time()))
        frame_dir = f"frames_{timestamp}"
        output_dir = f"detected_{timestamp}"
        output_video_path = f"output_with_lanes_{timestamp}.mp4"
        os.makedirs(frame_dir)
        os.makedirs(output_dir)

        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            input_video_path = tmp_file.name

        # Process the video
        with st.spinner("Processing video, please wait..."):
            success = process_video(input_video_path, frame_dir, output_dir, output_video_path)

        if success:
            # Read the processed video into bytes
            with open(output_video_path, "rb") as file:
                video_bytes = file.read()
            
            # Display the processed video
            st.write("Processed video with detected lanes:")
            st.video(video_bytes, format="video/mp4")
            
            # Provide download link using the same bytes
            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="output_with_lanes.mp4",
                mime="video/mp4"
            )

        # Clean up temporary files and directories
        shutil.rmtree(frame_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.remove(input_video_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)

if __name__ == "__main__":
    main()
