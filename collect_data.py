import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

file_name = 'hand_data.csv'

# Delete old file if it exists (Format is changing!)
if os.path.exists(file_name):
    print("WARNING: You should delete the old hand_data.csv manually to avoid mixing data formats.")
    # os.remove(file_name) # Uncomment to auto-delete

if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ['class_id']
        for i in range(21):
            headers.extend([f'x{i}', f'y{i}'])
        writer.writerow(headers)

cap = cv2.VideoCapture(0)

print("--- RELATIVE COORDINATE COLLECTOR ---")
print("Press '1': Hello | '2': ThumbsUp | '3': Victory | '4': Nothing")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y))
            wrist_x, wrist_y = landmarks[0]
            
            normalized_landmarks = []
            for x, y in landmarks:
                normalized_landmarks.extend([x - wrist_x, y - wrist_y])
            
            key = cv2.waitKey(1) & 0xFF
            class_id = -1
            if key == ord('1'): class_id = 0
            elif key == ord('2'): class_id = 1
            elif key == ord('3'): class_id = 2
            elif key == ord('4'): class_id = 3
            
            if class_id != -1:
                with open(file_name, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([class_id] + normalized_landmarks)
                print(f"Recorded: Class {class_id}")

    cv2.imshow('Collector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()