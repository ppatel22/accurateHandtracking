# Prince Patel
# 08/17/2023
# Run mediapipe hand tracking on webcam footage, calculate joint angles, then send to LUKEArm for mimicking
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import cv2
import os
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
import datetime
import threading

model_path = "/Users/princepatel/mit/accurateHandtracking/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
output_directory = "dataCombo"
handedness = None
palmTowards = None      # make sure this is updated and saved to the dataframe
os.makedirs(output_directory, exist_ok=True)


def extractCoordinatesL(
    result: HandLandmarkerResult,
    image: mp.Image,
    timestamp_ms: int,
):
    global finalL
    global handedness
    # Save the coordinates as a 2D numpy array with 21 columns, each containing the x, y, and z coordinates of a landmark
    hand_landmarks_list = result.hand_landmarks
    if hand_landmarks_list:
        handedness = result.handedness[0][0].display_name
        landmarks_data = []
        for norm in hand_landmarks_list[0]:
            landmarks_data.append(np.array([norm.x, norm.y, norm.z]))
        positions = calculateAngles(landmarks_data)
    else:
        positions = None
    if positions is not None:
        positions.append(timestamp_ms)
        finalL.loc[len(finalL)] = positions
    return None

def extractCoordinatesR(
    result: HandLandmarkerResult,
    image: mp.Image,
    timestamp_ms: int,
):
    global finalR
    # Save the coordinates as a 2D numpy array with 21 columns, each containing the x, y, and z coordinates of a landmark
    hand_landmarks_list = result.hand_landmarks
    if hand_landmarks_list:
        handedness = result.handedness[0][0].display_name
        landmarks_data = []
        for norm in hand_landmarks_list[0]:
            landmarks_data.append(np.array([norm.x, norm.y, norm.z]))
        positions = calculateAngles(landmarks_data)
    else:
        positions = None
    if positions is not None:
        positions.append(timestamp_ms)
        finalR.loc[len(finalR)] = positions
    return None

def calculateAngles(coordinates):
    # thumbAng = calculateThumb(coordinates)
    palmNormal = findPalm(coordinates)
    indexAng = calculateIndex(coordinates)
    middleAng = calculateMiddle(coordinates)
    (wristRot, wristFlex) = calculateWristAngles(normal=palmNormal)
    (thumbPAng, thumbYAng) = calculateThumbAngles(coordinates, normal=palmNormal)

    mrpAng = middleAng

    posCom = [
        thumbPAng,
        thumbYAng,
        indexAng,
        mrpAng,
        wristRot,
        wristFlex,
        0,
        0,
    ]  # build up the position command

    return posCom


def findPalm(coordinates):
    global palmTowards
    wrist = coordinates[0]
    index = coordinates[5]
    pinky = coordinates[17]

    vec1 = index - wrist
    vec2 = pinky - wrist
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(
        normal
    )  # defines the normal vector to the plane of the palm

    palmTowards = (
        True
        if (normal[2] > 0 and handedness == "Right")
        else (True if (normal[2] < 0 and handedness == "Left") else False)
    )

    return normal


def calculateThumb(coordinates):
    vec1 = coordinates[2] - coordinates[1]
    vec2 = coordinates[3] - coordinates[2]
    return angleBetweenVectors(vec1, vec2)


def calculateThumbAngles(coordinates, normal):
    wrist = coordinates[0]
    thumbCMC = coordinates[1]
    thumbMCP = coordinates[2]
    thumbIP = coordinates[3]
    index = coordinates[5]
    pinky = coordinates[17]

    vec01 = thumbCMC - wrist
    vec01 = vec01 / np.linalg.norm(vec01)
    vec23 = thumbIP - thumbMCP
    vec23 = vec23 / np.linalg.norm(vec23)

    thumbPAng = angleBetweenVectors(vec23, vec01)

    # find the plane running through normal and parallel to the index-pinky line
    indexPinky = index - pinky
    normH = np.cross(indexPinky, normal)
    normH = normH / np.linalg.norm(normH)
    thumbYProj = (
        -vec23 - np.dot(normH, -vec23) * normH
    )  # project the thumb angle onto that plane
    thumbYProj = thumbYProj if palmTowards else -thumbYProj

    # arctan of thumb projected angle gives thumb yaw
    thumbYAng = 90 - np.rad2deg(np.arctan2(thumbYProj[0], thumbYProj[2]))

    return (thumbPAng, thumbYAng)


def calculateIndex(coordinates):
    vec1 = coordinates[6] - coordinates[5]
    vec2 = coordinates[7] - coordinates[6]
    vec3 = coordinates[5] - coordinates[0]
    pipAng = angleBetweenVectors(vec1, vec2)
    mcpAng = angleBetweenVectors(vec1, vec3)

    return 0.5 * (pipAng + mcpAng)


def calculateMiddle(coordinates):
    vec1 = coordinates[10] - coordinates[9]
    vec2 = coordinates[11] - coordinates[10]
    vec3 = coordinates[9] - coordinates[0]

    pipAng = angleBetweenVectors(vec1, vec2)
    mcpAng = angleBetweenVectors(vec1, vec3)

    return 0.5 * (pipAng + mcpAng)


def calculateWristAngles(normal):
    rotAng = np.rad2deg(np.arctan2(normal[0], normal[2]))
    flexAng = np.rad2deg(np.arcsin(-normal[1]))

    return (rotAng, flexAng)


@staticmethod
def angleBetweenVectors(v1, v2, alternate=False):
    angle = np.rad2deg(
        np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        )
    )
    if alternate:
        angle = -angle
    return angle


landmarker_optionsL = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=extractCoordinatesL,
)

landmarker_optionsR = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=extractCoordinatesR,
)

finalL = pd.DataFrame(
    columns=[
        "Angle L1",
        "Angle L2",
        "Angle L3",
        "Angle L4",
        "Angle L5",
        "Angle L6",
        "Angle L7",
        "Angle L8",
        "Timestamp L (ms)",
    ]
)
finalR = pd.DataFrame(
    columns=[
        "Angle R1",
        "Angle R2",
        "Angle R3",
        "Angle R4",
        "Angle R5",
        "Angle R6",
        "Angle R7",
        "Angle R8",
        "Timestamp R (ms)",
    ]
)


def process_camera(cap, landmarker, camera_index):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from webcam {camera_index}.")
            break
        # cv2.imshow(f"Live Feed {camera_index}", frame)

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Send live image data to perform hand landmarks detection.
        frame_timestamp_ms = int(round(time.time() * 1000))
        result = landmarker.detect_async(mp_image, frame_timestamp_ms)
        print(f"Handedness: {handedness}, Palm Towards: {palmTowards}")
    return None


# Create threads for each camera
cap1 = cv2.VideoCapture(0)  # Use the appropriate camera index for the first camera
cap2 = cv2.VideoCapture(1)  # Use the appropriate camera index for the second camera

# Create a separate HandLandmarker instance for each camera
landmarkerL = mp.tasks.vision.HandLandmarker.create_from_options(landmarker_optionsL)
landmarkerR = mp.tasks.vision.HandLandmarker.create_from_options(landmarker_optionsR)

# Start threads for each camera
thread1 = threading.Thread(target=process_camera, args=(cap1, landmarkerL, 0))
thread2 = threading.Thread(target=process_camera, args=(cap2, landmarkerR, 1))

# Start the threads
thread1.start()
thread2.start()

try:
    while True:
        pass
except KeyboardInterrupt:
    file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    file_path = os.path.join(output_directory, file_name)
    finalcombo = pd.concat([finalL, finalR], axis=1)
    finalcombo.to_csv(file_path)

    # Release camera resources
    cap1.release()
    cap2.release()

    # Close OpenCV windows
    cv2.destroyAllWindows()

    # Wait for threads to finish
    thread1.join()
    thread2.join()
"""
NOTES:
- result of landmarker.detect_async(mp_image, frame_timestamp_ms) is a list of HandLandmarkerResult objects (https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/components/containers/landmark_detection_result.py)
- Each landmark can either be normalized or pixel coordinates. (https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/components/containers/landmark.py)
"""

"""
Notes from 9/26/23:
The trackerL.py file should work perfectly fine now. Instead of running two copies of the same program (one for each camera), 
I am trying to use threading in this combined tracker program. The program runs in its current state, as I have made minmal 
changes. Basically, I moved the camera frame reading and processing to a reusable function. Now, I need to create a new dataframe 
that can save data from both cameras. This will most likely involve changing or duplicating the extractCoordinates function.
"""
