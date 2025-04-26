import mediapipe as mp
import cv2
import numpy as np

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image,model):
    """
    Let us to start the detection of nthe keypoints in a frame

    ARGS:
        image: the frame where the keypoints will be detect
        model: the model used for the detection

    Returns:
        image: the frame used for the detection
        results: the model's keypoints predicted
    """

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image,results):
    """
    Draw the conection in the pose, left hands and righ hand to see the keypoints in the frame

    ARGS:

        Image: the frame took it for the webcam
        results: the keypoints predicted in the frame
    """

    # POSE CONNECTIONS
    mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(128,53,128),thickness=2,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1))
    
    # LEFT HAND CONNECTIONS
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(175,30,100),thickness=2,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(175,110,25),thickness=2,circle_radius=1))
    
    # RIGHT HAND CONNECTIONS
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(40,155,100),thickness=2,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,155,40),thickness=2,circle_radius=1))
    
def extract_keypoints(results):
    """
    ARGS:

        results: the results obtained from ther prediction of the holicti model

    Returns:
        Vector with the pose, left hand, right hand landmarks
    """

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose,lh,rh]),lh,rh

"""
def prob_viz(res,actions,input_frame,colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame,(0,60+num*40),(int(prob*100),90+num*40), colors[num],-1)
        cv2.putText(output_frame,actions[num],(0,85+num*40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    return output_frame
"""

def draw_hand_box(res,lh,rh,input_frame, color=(50,0,255),thickness=2,padding=15):
    domain_expansion = ["Unlimited Void", "Idle Death Gamble", "Chimera Shadow Garden",
                        "Malevolant Shrine","Unknown name","Authentic Mutual Love"  ]
    
    character_domain = domain_expansion[np.argmax(res)]
    h,w,_ = input_frame.shape
    output_frame = input_frame.copy()
    if np.count_nonzero(lh) == 0 and np.count_nonzero(rh) == 0:
        return input_frame
    
    # Left hand coordinates
    if np.count_nonzero(lh) > 0:
        lh_coords = lh.reshape(21,3)
        lh_x_coords = [int(x*w) for x in lh_coords[:,0]]
        lh_y_coords = [int(y*h) for y in lh_coords[:,1]]
        lh_x_min = min(lh_x_coords)
        lh_x_max = max(lh_x_coords)
        lh_y_min = min(lh_y_coords)
        lh_y_max = max(lh_y_coords)
    else:
        lh_x_min = lh_x_max = lh_y_min = lh_y_max = 0

    # Reft hand coordinates
    if np.count_nonzero(rh) > 0:
        rh_coords = rh.reshape(21,3)
        rh_x_coords = [int(x*w) for x in rh_coords[:,0]]
        rh_y_coords = [int(y*h) for y in rh_coords[:,1]]
        rh_x_min = min(rh_x_coords)
        rh_x_max = max(rh_x_coords)
        rh_y_min = min(rh_y_coords)
        rh_y_max = max(rh_y_coords)
    else:
        rh_x_min = rh_x_max = rh_y_min = rh_y_max = 0

    # When only one hand is detected
    if np.count_nonzero(lh) == 0:
        lh_x_min, lh_x_max, lh_y_min, lh_y_max = rh_x_min, rh_x_max, rh_y_min, rh_y_max
    elif np.count_nonzero(rh) == 0:
        rh_x_min, rh_x_max, rh_y_min, rh_y_max = lh_x_min, lh_x_max, lh_y_min, lh_y_max

    # Global coordinates

    x_min = max(min(lh_x_min,rh_x_min)-padding,0)
    x_max = min(max(lh_x_max,rh_x_max)+padding,w)
    y_min = max(min(lh_y_min,rh_y_min)-padding,0)
    y_max = min(max(lh_y_max,rh_y_max)-padding,h)

    cv2.rectangle(output_frame,(x_min,y_min),(x_max,y_max),color=color,thickness=thickness)
    text_size = cv2.getTextSize(character_domain, cv2.FONT_HERSHEY_SIMPLEX, 0.8,2)[0]
    text_x = (x_min + x_max - text_size[0]) // 2
    text_y = y_min - 10

    text_y = max(text_y,30)
    cv2.putText(output_frame, character_domain, (text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)


    return output_frame


