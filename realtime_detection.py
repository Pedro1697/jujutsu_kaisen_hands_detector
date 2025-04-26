import cv2
import mediapipe as mp
from auxiliar_functions_landmarks import  *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
characters = ["Gojo", "Hakari", "Megumi","Sukuna","Yuji","Yuta"]
colors = [(245,117,16),(117,245,16),(16,117,245),(245,117,16),(117,245,16),(16,117,245)]

print("Loading model")

model = Sequential(
                   [LSTM(64,return_sequences=False,activation="relu",input_shape=(30,258)),
                    Dense(64,activation="relu"),
                    Dense(32,activation="relu"),
                    Dense(6,activation="softmax")]
                  )

model.load_weights("/Users/pedronicolas/Desktop/jjk_classifier/best_weigths.weights.h5")

print("Loading model complete... Starting predictions!")

sequence = []
predictions = []
sentence = []
threshold = 0.7



cap = cv2.VideoCapture(1)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)


        # Prediction logic
        keypoints,lh,rh = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res =  model.predict(np.expand_dims(sequence,axis=0))[0]
            print(characters[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Visualization logic

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if characters[np.argmax(res)] != sentence[-1]:
                            sentence.append(characters[np.argmax(res)])
                        else:
                            sentence.append(characters[np.argmax(res)])
            if len(sentence) > 5:
                sentence = sentence[-5:]
        
            # Visualization prob

            #image = prob_viz(res,characters,image,colors)
            image = draw_hand_box(res,lh,rh,image)

        #cv2.rectangle(image,(0,0),(640,40),(245,117,16),-1)
        #cv2.putText(image,' '.join(sentence),(3,30),
                    #cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    