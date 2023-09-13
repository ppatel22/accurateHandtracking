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

from joints1 import jointAngles1

model_path = "/Users/princepatel/mit/accurateHandtracking/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
output_directory = "data"
handedness = None
palmTowards = None
os.makedirs(output_directory, exist_ok=True)


def extractCoordinates(
    result: HandLandmarkerResult,
    image: mp.Image,
    timestamp_ms: int,
):
    # Save the coordinates as a 2D numpy array with 21 columns, each containing the x, y, and z coordinates of a landmark
    hand_landmarks_list = result.hand_landmarks
    if hand_landmarks_list:
        handedness = result.handedness[0][0].display_name
        landmarks_data = []
        for norm in hand_landmarks_list[0]:
            landmarks_data.append(np.array([norm.x, norm.y, norm.z]))
        print(landmarks_data)
        positions = calculateAngles(landmarks_data)
    else:
        positions = None
    if positions is not None:
        positions.append(timestamp_ms)
        final.loc[len(final)] = positions
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


landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=extractCoordinates,
)

final = pd.DataFrame(columns=["Angle 1", "Angle 2", "Angle 3", "Angle 4", "Angle 5", "Angle 6", "Angle 7", "Angle 8", "Timestamp (ms)"])
try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        landmarker_options
    ) as landmarker:
        # Use OpenCV’s VideoCapture to start capturing from the webcam.
        cap = cv2.VideoCapture(1)  # Use the appropriate camera index if not the default
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from webcam.")
                break
            cv2.imshow("Live Feed", frame)

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # Send live image data to perform hand landmarks detection.
            frame_timestamp_ms = int(round(time.time() * 1000))
            result = landmarker.detect_async(mp_image, frame_timestamp_ms)
            print("result saved", frame_timestamp_ms)
except KeyboardInterrupt:
    file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    file_path = os.path.join(output_directory, file_name)
    final.to_csv(file_path)
    cap.release()
    cv2.destroyAllWindows()
"""
NOTES:
- result of landmarker.detect_async(mp_image, frame_timestamp_ms) is a list of HandLandmarkerResult objects (https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/components/containers/landmark_detection_result.py)
- Each landmark can either be normalized or pixel coordinates. (https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/components/containers/landmark.py)
"""
