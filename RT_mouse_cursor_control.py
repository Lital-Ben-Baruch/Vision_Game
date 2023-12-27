import cv2
import mediapipe as mp
import pyautogui
import time
from tkinter import *
from PIL import Image, ImageTk

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Variables to track hand stability
last_hand_position = None
hand_stable_start_time = None
hand_stable_duration = 1  # Duration in seconds for the hand to be considered stable
movement_threshold = 0.06  # Threshold for detecting hand movement

# Create a tkinter window
root = Tk()
root.title("Hand Detection")
root.geometry('320x240')  # Set initial window size
root.resizable(True, True)  # Allow the window to be resizable
root.attributes('-topmost', True)  # Keep the window always on top

# Create a label in the tkinter window to show the video frames
label = Label(root)
label.pack(fill=BOTH, expand=YES)

# Function to display the video frames and handle hand tracking
def show_frames():
    global last_hand_position, hand_stable_start_time

    ret, frame = cap.read()
    if not ret:
        root.destroy()
        return

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Resize the frame to fit the label size
    frame = cv2.resize(frame, (label.winfo_width(), label.winfo_height()))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        wrist_landmark = hand_landmarks.landmark[0]
        hand_x = int(wrist_landmark.x * frame.shape[1])
        hand_y = int(wrist_landmark.y * frame.shape[0])

        # Convert hand position to screen coordinates
        screen_x = int(hand_x * screen_width / frame.shape[1])
        screen_y = int(hand_y * screen_height / frame.shape[0])

        pyautogui.moveTo(screen_x, screen_y)

        current_hand_position = (wrist_landmark.x, wrist_landmark.y)

        if last_hand_position:
            distance = ((current_hand_position[0] - last_hand_position[0]) ** 2 +
                        (current_hand_position[1] - last_hand_position[1]) ** 2) ** 0.5
            if distance < movement_threshold:
                if not hand_stable_start_time:
                    hand_stable_start_time = time.time()
                elif time.time() - hand_stable_start_time >= hand_stable_duration:
                    pyautogui.click()
                    print(f"Mouse clicked at ({screen_x}, {screen_y})")
                    hand_stable_start_time = None
            else:
                hand_stable_start_time = None

        last_hand_position = current_hand_position

    # Convert the frame to a format suitable for tkinter
    cv_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=cv_img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, show_frames)

# Start the video in tkinter
show_frames()

# Run the tkinter loop
root.mainloop()

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
