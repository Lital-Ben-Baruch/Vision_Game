import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Variables to track hand stability
last_hand_position = None
hand_stable_start_time = None
hand_stable_duration = 3  # Duration in seconds for the hand to be considered stable
movement_threshold = 0.05  # Threshold for detecting hand movement

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(frame_rgb)

    # Draw the hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the wrist landmark (landmark 0) position
            wrist_landmark = hand_landmarks.landmark[0]
            hand_x = int(wrist_landmark.x * frame.shape[1])
            hand_y = int(wrist_landmark.y * frame.shape[0])

            # Convert hand position to screen coordinates
            screen_x = int(hand_x * screen_width / frame.shape[1])
            screen_y = int(hand_y * screen_height / frame.shape[0])

            # Move the mouse cursor to the hand's position
            pyautogui.moveTo(screen_x, screen_y)

            current_hand_position = (wrist_landmark.x, wrist_landmark.y)

            # Check if the hand is in the same position
            if last_hand_position:
                distance = ((current_hand_position[0] - last_hand_position[0]) ** 2 +
                            (current_hand_position[1] - last_hand_position[1]) ** 2) ** 0.5
                if distance < movement_threshold:
                    if not hand_stable_start_time:
                        hand_stable_start_time = time.time()
                    elif time.time() - hand_stable_start_time >= hand_stable_duration:
                        # Trigger a mouse click
                        pyautogui.click()
                        print(f"Mouse clicked at ({screen_x}, {screen_y})")

                        # Reset the timer
                        hand_stable_start_time = None
                else:
                    hand_stable_start_time = None

            last_hand_position = current_hand_position

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
