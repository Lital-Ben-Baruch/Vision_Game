import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

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

            # Perform click at the hand's position on the screen
            pyautogui.click(screen_x, screen_y)

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
