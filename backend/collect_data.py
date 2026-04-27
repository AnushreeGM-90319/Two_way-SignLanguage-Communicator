import cv2
import numpy as np
import os
import mediapipe as mp

# CONFIGURATION
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no', 'help', 'sorry', 'please', 'okay'])
no_sequences = 120
sequence_length = 30

# MEDIAPIPE SETUP
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    # Upper body pose indices
    upper_body_indices = [0, 11, 12, 13, 14, 15, 16]

    # Pose
    if results.pose_landmarks:
        pose = np.array([
            [results.pose_landmarks.landmark[i].x,
             results.pose_landmarks.landmark[i].y,
             results.pose_landmarks.landmark[i].z,
             results.pose_landmarks.landmark[i].visibility]
            for i in upper_body_indices
        ]).flatten()
    else:
        pose = np.zeros(len(upper_body_indices) * 4)
    #Left Hand
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    #Right Hand
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            # Check if the last frame (29.npy) of this sequence already exists
            check_path = os.path.join(DATA_PATH, action, str(sequence), f"{sequence_length-1}.npy")
            if os.path.exists(check_path):
                print(f"Skipping {action} video {sequence} (Already recorded)")
                continue # Skip to the next video
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

            # --- WAIT FOR USER TO PRESS 'S' ---
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, f'ACTION: {action} | Video #{sequence}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'READY? Press "S" to Start Recording', (80,230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', frame)
                
                char = cv2.waitKey(1) & 0xFF
                if char == ord('s'): 
                    break
                if char == ord('q'): 
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Data collection stopped by user.")
                    exit()

            # --- RECORD THE 30 FRAMES ---
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                keypoints = extract_landmarks(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                cv2.putText(frame, f'RECORDING: {action} Frame {frame_num}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1)
                    
    cap.release()
    cv2.destroyAllWindows()
    print("All data collection complete!")