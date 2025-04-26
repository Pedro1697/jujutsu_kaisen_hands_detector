import numpy as np
import mediapipe as mp
from auxiliar_functions_landmarks import *
import cv2
import os

IMAGE_PATH = "/Users/pedronicolas/Desktop/jjk_classifier/image_dataset2"

#characters = ["Gojo", "Hakari", "Megumi","Sukuna","Yuji","Yuta"]
characters =["Megumi"]
num_sequences = 30
sequence_length = 30


cap = cv2.VideoCapture(1)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    for character in characters:
        for sequence in range(num_sequences):
            character_folder = os.path.join(IMAGE_PATH,character,str(sequence))
            os.makedirs(character_folder,exist_ok=True)
            for n_frame in range(sequence_length):

              # Read feed
              ret, frame = cap.read()

              # Make detections
              image, results = mediapipe_detection(frame, holistic)
              #print(results)

              # Draw landmarks
              draw_styled_landmarks(image, results)

              if n_frame == 0:
                  cv2.putText(image,"STARTING COLLECTION",(120,200),
                              cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                  cv2.putText(image,f"Collecting frame for {character} in the video number {sequence}",(15,12),
                              cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                  # Show to screen
                  cv2.imshow('OpenCV Feed', image)
                  cv2.waitKey(2000)
              else:
                  cv2.putText(image,f"Collecting frame for {character} in the video number {sequence}",(15,12),
                              cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                  # Show to screen
                  cv2.imshow('OpenCV Feed', image)
                
              keypoints = extract_keypoints(results)
              npy_path = os.path.join(character_folder,str(n_frame)) 
              np.save(npy_path,keypoints)
                

              # Break gracefully
              if cv2.waitKey(100) & 0xFF == ord('q'):
                  break
    cap.release()
    cv2.destroyAllWindows()
