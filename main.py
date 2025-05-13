import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import time
import pyttsx3

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


engine = pyttsx3.init()
def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False)

cap = cv2.VideoCapture(0)


word = ""
last_prediction = ""
stable_start_time = None
state = "waiting"
last_detected_time = time.time()
pause_timeout = 2.0 
confirmation_time = 1.0 

confirmed_letter = ""
confirmed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    current_time = time.time()

    if results.multi_hand_landmarks:
        last_detected_time = current_time
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        feature = []
        for lm in hand_landmarks.landmark:
            feature.extend([lm.x, lm.y, lm.z])

        if len(feature) == 63:
            feature_scaled = scaler.transform([feature])
            prediction = model.predict(feature_scaled)[0]

            if prediction != last_prediction:
                last_prediction = prediction
                stable_start_time = current_time
                confirmed = False
            else:
                if stable_start_time and (current_time - stable_start_time) >= confirmation_time and not confirmed:
                    confirmed_letter = prediction
                    word += confirmed_letter
                    confirmed = True
                    state = "confirmed"

    else:
        if word and (current_time - last_detected_time) > pause_timeout:
            state = "pronounce"
            speak(word)
            time.sleep(1.5)
            word = ""
            confirmed_letter = ""
            last_prediction = ""
            stable_start_time = None
            confirmed = False
            state = "waiting"


    if state == "confirmed":
        cv2.putText(frame, f"Confirmed: {confirmed_letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Current word: {word}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if state == "pronounce":
        cv2.putText(frame, f"Speaking: {word}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow('ASL Translator', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
