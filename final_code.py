import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import load_model

#creating variables 1.for holistic 2.for drawing utilities
mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utilities

actions = np.array(['how','I','listen','neutral','no','speak','who','work','yes','you']) #actions that we try to detect
no_sequences = 20 #20 videos worth of data
sequence_length = 30 #each video has 30 frames in length

#function for mediapipe detection
def mediapipe_detection(image, model):
    #opencv reads in bgr but mediapipe uses rgb
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #conversion from bgr to rgb
    image.flags.writeable = False #image no longer writable
    results = model.process(image) #dectecting using mediapipe/making prediction
    image.flags.writeable = True #image now writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #conversion from rgb to bgr
    return image,results

#custom draw landmarks(later)
def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )#draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )#draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )#draw right hand connections

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

#real-time test[
sequence = []
sentence = []
threshold = 0.98 #how confident you want
# Track the repetition of predictions
repetition_count = {}
current_action = None
required_reps = 8  # how many times in a row before accepting the prediction

#load the model
model = load_model('the_model_final.keras')

#capture
cap = cv2.VideoCapture(0)
#set mediapipe model 1.detect first 2.track from detected keypoints
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #make detections
        image,results = mediapipe_detection(frame,holistic)

        #draw landmarks
        draw_styled_landmarks(image,results) #send that frame(image) to draw landmarks

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:] #grab the last 60 frames

        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence,axis=0))[0] #initially it was (30,1662) but it needs (x,30,1662)
            print(actions[np.argmax(res)]) #print the most confident one
            print(res[np.argmax(res)])
            # cv2.rectangle(image, (0, 0), (700, 120), (245, 117, 16), -1)

            if res[np.argmax(res)] >= threshold:
                # If it's the same action as previous, increase count
                if actions[np.argmax(res)] == current_action:
                    repetition_count[actions[np.argmax(res)]] = repetition_count.get(actions[np.argmax(res)] , 0) + 1
                else:
                    # Reset count and switch to new action
                    repetition_count = {actions[np.argmax(res)] : 1}
                    current_action = actions[np.argmax(res)]
            # Confirm action only after repeated predictions
            if repetition_count.get(actions[np.argmax(res)], 0) >= required_reps:
                if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                if actions[np.argmax(res)] == 'neutral':
                    sentence.clear()
        if len(sentence) > 5:
            sentence = sentence[-5:]
        cv2.rectangle(image,(0,0),(850,120),(245,117,16),-1)
        cv2.putText(image,' '.join(sentence),(3,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed',image) #show that image(the one sent and processed with landmarks)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#]