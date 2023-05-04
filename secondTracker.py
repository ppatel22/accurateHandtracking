# Mikey Fernandez
# 11/29/2022
# Run mediapipe hand tracking on webcam footage, calculate joint angles, then send to LUKEArm for mimicking

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sys
import threading

import time
import datetime

import lcm
from joints2 import jointAngles2

class handTracker():
    """The 21 hand landmarks."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self, camera=1, frameWidth=None, frameHeight=None):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.handedness = None
        self.palmTowards = True

        self.camera = camera # webcam is 0, wired input is 1

        self.numJoints = 8
        self.posCom = np.zeros(self.numJoints)

        cap = cv2.VideoCapture(self.camera)
        if frameWidth is not None: cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
        if frameHeight is not None: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
        self.frameRate = cap.get(cv2.CAP_PROP_FPS) # 30 FPS
        self.cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.filters = BesselFilterArr(numChannels=self.numJoints, order=4, critFreqs=[14, 14, 14, 14, 14, 14, 14, 14], fs=self.frameRate, filtType='lowpass')
        self.history = np.zeros((self.numJoints, int(self.frameRate)))
        self.cap = cap

        self.startTime = time.time()

        # for saving
        lmkNames = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
        self.colNames = sum(list(map(lambda lmk: [lmk + "_X", lmk + "_Y", lmk + "_Z"], lmkNames)), []) # this is a stupid way to get X, Y, Z added to the names and collapse into a list
        self.colNames.extend(["thumbPPos", "thumbYPos", "indexPos", "mrpPos", "wristRot", "wristFlex", "humPos", "elbowPos"])

        self.output = np.empty((1, len(self.colNames)))

    def extractCoordinates(self, hand_landmarks):
        grouped = [[lmk.x*self.cols, lmk.y*self.rows, lmk.z*self.cols] for lmk in hand_landmarks.landmark]
        coordinates = np.asarray(grouped)

        return coordinates

    def calculateAngles(self, coordinates):
        # thumbAng = self.calculateThumb(coordinates)
        palmNormal = self.findPalm(coordinates)
        indexAng = self.calculateIndex(coordinates)
        middleAng = self.calculateMiddle(coordinates)
        (wristRot, wristFlex) = self.calculateWristAngles(normal=palmNormal)
        (thumbPAng, thumbYAng) = self.calculateThumbAngles(coordinates, normal=palmNormal)

        mrpAng = middleAng
        
        posCom = [thumbPAng, thumbYAng, indexAng, mrpAng, wristRot, wristFlex, 0, 0] # build up the position command

        return posCom

    def findPalm(self, coordinates):
        wrist = coordinates[0]
        index = coordinates[5]
        pinky = coordinates[17]

        vec1 = index - wrist
        vec2 = pinky - wrist
        normal = np.cross(vec1, vec2); normal = normal/np.linalg.norm(normal) # defines the normal vector to the plane of the palm

        self.palmTowards = True if (normal[2] > 0 and self.handedness == "Right") else (True if (normal[2] < 0 and self.handedness == "Left") else False) 

        return normal

    def calculateThumb(self, coordinates):
        vec1 = coordinates[2] - coordinates[1]
        vec2 = coordinates[3] - coordinates[2]
        return self.angleBetweenVectors(vec1, vec2)

    def calculateThumbAngles(self, coordinates, normal):
        wrist = coordinates[0]
        thumbCMC = coordinates[1]
        thumbMCP = coordinates[2]
        thumbIP = coordinates[3]
        index = coordinates[5]
        pinky = coordinates[17]

        vec01 = thumbCMC - wrist; vec01 = vec01/np.linalg.norm(vec01)
        vec23 = thumbIP - thumbMCP; vec23 = vec23/np.linalg.norm(vec23)

        thumbPAng = self.angleBetweenVectors(vec23, vec01)

        # find the plane running through normal and parallel to the index-pinky line
        indexPinky = index - pinky
        normH = np.cross(indexPinky, normal); normH = normH/np.linalg.norm(normH)
        thumbYProj = -vec23 - np.dot(normH, -vec23)*normH # project the thumb angle onto that plane
        thumbYProj = thumbYProj if self.palmTowards else - thumbYProj

        # arctan of thumb projected angle gives thumb yaw
        thumbYAng = 90 - np.rad2deg(np.arctan2(thumbYProj[0], thumbYProj[2]))

        return (thumbPAng, thumbYAng)

    def calculateIndex(self, coordinates):
        vec1 = coordinates[6] - coordinates[5]
        vec2 = coordinates[7] - coordinates[6]
        vec3 = coordinates[5] - coordinates[0]
        pipAng = self.angleBetweenVectors(vec1, vec2)
        mcpAng = self.angleBetweenVectors(vec1, vec3)

        return 0.5*(pipAng + mcpAng)

    def calculateMiddle(self, coordinates):
        vec1 = coordinates[10] - coordinates[9]
        vec2 = coordinates[11] - coordinates[10]
        vec3 = coordinates[9] - coordinates[0]

        pipAng = self.angleBetweenVectors(vec1, vec2)
        mcpAng = self.angleBetweenVectors(vec1, vec3)

        return 0.5*(pipAng + mcpAng)

    def calculateWristAngles(self, normal):
        rotAng = np.rad2deg(np.arctan2(normal[0], normal[2]))            
        flexAng = np.rad2deg(np.arcsin(-normal[1]))

        return (rotAng, flexAng)

    @staticmethod
    def angleBetweenVectors(v1, v2, alternate=False):
        angle = np.rad2deg(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)))
        if alternate:
            angle = -angle
        return angle


    def sendMessage(self, angles):
        msg = jointAngles2()
        msg.timestamp = self.startTime
        msg.angle0 = angles[0]
        msg.angle1 = angles[1]
        msg.angle2 = angles[2]
        msg.angle3 = angles[3]
        msg.angle4 = angles[4]
        msg.angle5 = angles[5]
        msg.angle6 = angles[6]
        msg.angle7 = angles[7]
        lc = lcm.LCM()
        lc.publish("secondTracker", msg.encode())
        return None

    def processImage(self, inImage):
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image = inImage.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results = self.hands.process(image)
        annotated_image = image.copy()

        annotated_image.flags.writeable = True
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            self.handedness = results.multi_handedness[0].classification[0].label
            for hand_landmarks in results.multi_hand_landmarks:
                coords = self.extractCoordinates(hand_landmarks)

                posCom = self.calculateAngles(coords)
                # posCom = self.applyFilter(posCom)
                # The line below prints out the timestamp and coordinates in the terminal.
                # print(f'{(time.time() - self.startTime):.5f}', [f'{pos:07.3f}' for pos in posCom])

                # The line below uses a hleper function to send the timestamp and angles via LCM
                self.sendMessage(posCom)
                
                self.mp_drawing.draw_landmarks(annotated_image, hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                self.output = np.concatenate((self.output, np.append(coords, posCom)[None, :]), axis=0)
                
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Annotated Hand', annotated_image)

            return posCom

        return np.zeros(self.numJoints)

    def runTracking(self, arm=None):
        while self.cap.isOpened():
            success, image = self.cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            # do image processing separately here
            posCom = self.processImage(image)
            # if arm is not None: self.sendToArm(arm=arm, posCom=posCom)

            # quit condition - the ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()

    # def saveData(self, saveName=None):
    #     outputDf = pd.DataFrame(data=self.output, columns=self.colNames)
    #     curTime = str(datetime.datetime.now()).split('.')[0] # removes microseconds
    #     curTime = curTime.replace(" ", "_")
    #     if saveName is None:
    #         outputDf.to_csv('./data/handTrackingCoordinates_' + str(curTime) + '.csv', encoding='utf-8', sep="\t", index=False)
    #     else:
    #         outputDf.to_csv('./' + saveName + '.csv', encoding='utf-8', sep="\t", index=False)

    def analyzeVideo(self, fileName):
        vidFile = cv2.VideoCapture(fileName)
        length = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
        frameRate = np.ceil(vidFile.get(cv2.CAP_PROP_FPS))
        resolution = (int(vidFile.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidFile.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        (self.cols, self.rows) = resolution

        print(f'Input video information: {length} frames ({frameRate} FPS) captured at {resolution}')

        with self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            for frameNum in range(length):
                status, frame = vidFile.read()

                if not status:
                    print(f'Frame number {frameNum} not read correctly')
                    continue

                # Read an image, flip it around y-axis for correct handedness output (see above).
                image = cv2.flip(frame, 1)

                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    continue

                for hand_landmarks in results.multi_hand_landmarks:
                    coords = self.extractCoordinates(hand_landmarks)
                    posCom = self.calculateAngles(coords)

                    self.output = np.concatenate((self.output, np.append(coords, posCom)[None, :]), axis=0)

                # Draw hand world landmarks.
                if not results.multi_hand_world_landmarks:
                    continue

                if not frameNum % 100:
                    print(f'Processed frame {frameNum} of {length}')

        print(f'Done - saving data to ./data/{fileName[:-4]}.csv')
        # self.saveData(saveName='data/' + fileName[:-4])

if __name__ == '__main__':
    print('Starting hand tracker')
    tracker = handTracker()
    sendingToArm = False

    tracker.runTracking(arm=None)
    # if saving: tracker.saveData()

    tracker1 = handTracker(camera=0, frameWidth=480, frameHeight=320)

    thread1 = threading.Thread(target=tracker1.runTracking, name='Camera1')

    print('Done.')