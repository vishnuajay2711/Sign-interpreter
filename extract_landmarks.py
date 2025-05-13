import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

dataset_path = 'asl_alphabet_train/asl_alphabet_train'
data = []

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    print(f'Processing label: {label}')
    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            data.append((landmarks, label))

with open('landmarks.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f'Extracted {len(data)} samples.')
