"""
extract_frames_from_videos.py

This script extracts and saves individual frames from all video files located in a specified directory.
Each video's frames are saved in a separate folder named after the video file.
The files name will be saved as 'videoFileName_FrameNumber'
Prerequisites:
- Python installed with OpenCV library (opencv-python package).
- Videos should be in .mp4 format (or modify the script to match your video format).
- Ensure the specified directory path is correct and accessible.
- Write permissions are required in the directory.

Usage:
- Set the 'video_dir' variable to the path of the directory containing your video files.
- Run the script. It will create subdirectories within the specified directory and save the frames there.
"""

import cv2
import os
import glob

# Directory containing the video files
video_dir = 'C:/Users/lital/Desktop/VisionProject/phone/22_11_2023/sub_videos/hand'

# Iterate over each video file in the directory
for video_file in glob.glob(video_dir + '/*.mp4'):  # Modify for different video formats
    # Extract the base name of the video file
    base_name = os.path.basename(video_file)
    base_name_no_ext = os.path.splitext(base_name)[0]

    # Create a directory for frames of this video
    frame_dir = os.path.join(video_dir, base_name_no_ext + '_frames')
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Save each frame as an image
        frame_file = os.path.join(frame_dir, f'{base_name_no_ext}_{frame_number}.jpg')
        cv2.imwrite(frame_file, frame)
        frame_number += 1

    # Release the video capture object
    cap.release()

print("All frames extracted and saved.")
