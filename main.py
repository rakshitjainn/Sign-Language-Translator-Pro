import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import threading

def speak_now(text):
    def run_voice():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Voice Error: {e}")
            
    thread = threading.Thread(target=run_voice)
    thread.start()

try:
    with open('sign_language_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model not found. Please run train_model.py.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

labels_dict = {0: 'Hello', 1: 'Thumbs Up', 2: 'Victory', 3: 'Nothing'}
cap = cv2.VideoCapture(0)
current_prediction = "Nothing"
stable_prediction = "Nothing"
frame_counter = 0
FRAME_THRESHOLD = 5
print("--- SYSTEM READY ---")
print("Show a sign!")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_prediction = "Nothing"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            x_raw = []
            y_raw = []
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y))
                x_raw.append(lm.x)
                y_raw.append(lm.y)
            
            wrist_x, wrist_y = landmarks[0]
            normalized_landmarks = []
            for x, y in landmarks:
                normalized_landmarks.extend([x - wrist_x, y - wrist_y])

            prediction_proba = model.predict_proba([normalized_landmarks])[0]
            max_proba = np.max(prediction_proba)
            predicted_index = np.argmax(prediction_proba)
            x_min = int(min(x_raw) * W)
            y_min = int(min(y_raw) * H) - 10
            if max_proba > 0.8:
                temp_label = labels_dict[predicted_index]
                if temp_label != "Nothing":
                    frame_prediction = temp_label
                    cv2.putText(frame, f"{temp_label}", (x_min, y_min), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if frame_prediction == current_prediction:
        frame_counter += 1
    else:
        current_prediction = frame_prediction
        frame_counter = 0

    if frame_counter > FRAME_THRESHOLD:
        if current_prediction != stable_prediction:
            stable_prediction = current_prediction
            if stable_prediction != "Nothing":
                speak_now(stable_prediction)
    cv2.imshow('Final Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()