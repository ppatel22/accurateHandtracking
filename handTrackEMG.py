# Mikey Fernandez
# 12/07/2022
# Record EMG and Mediapipe hand angles simultaneously
# TODO: get these in the right form for model training

import sys
sys.path.append('/home/haptix/haptix/haptix_controller/handsim/src/')
import os

import numpy as np
import pandas as pd
import cv2
import datetime
from trackerClass import handTracker
from EMGClass import EMG
from time import sleep

class EMG_hand_recorder():
    def __init__(self, mainFol, folName='.'):
        self.tracker = handTracker()
        self.emg = EMG()
        self.emg.startCommunication()
        self.mainFol = mainFol

        self.folName = folName

        self.posHistory = np.empty((1, self.tracker.numJoints))
        self.emgHistory = np.empty((1, self.emg.numElectrodes))

    def trackAndRecord(self, saveType='normed'):
        # cap = self.tracker.cap
        while self.tracker.cap.isOpened():
            success, image = self.tracker.cap.read()

            if saveType == 'normed':
                thisEMG = np.asarray(self.emg.normedEMG)
            elif saveType == 'iEMG':
                thisEMG = np.asarray(self.emg.iEMG)
            elif saveType == 'act':
                thisEMG = np.asarray(self.emg.muscleAct)
            else:
                raise ValueError(f'Improper EMG saving type {saveType}')

            if not success:
                print("Ignoring empty camera frame.")
                continue

            # do image processing separately here
            posCom = self.tracker.processImage(image)
            posCom = np.deg2rad(posCom)

            # quit condition - the ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # add the recorded values to the array for saving
            self.posHistory = np.concatenate((self.posHistory, posCom[None, :]), axis=0)
            self.emgHistory = np.concatenate((self.emgHistory, thisEMG[None, :]), axis=0)

        self.tracker.cap.release()
        self.emg.exitEvent.set()

        # now save the data
        self.saveData()

    def saveData(self):
        posDf = pd.DataFrame(data=self.posHistory, columns=["thumbPPos", "thumbYPos", "indexPos", "mrpPos", "wristRot", "wristFlex", "humPos", "elbowPos"])
        emgDf = pd.DataFrame(data=self.emgHistory)

        runFol = os.getcwd()
        curTime = str(datetime.datetime.now()).split('.')[0] # removes microseconds
        curTime = curTime.replace(" ", "_")
        saveFolPath = os.path.join(runFol, self.mainFol, self.folName + '_' + curTime)

        try:
            os.mkdir(saveFolPath)
        except FileExistsError as e:
            pass

        posDf.to_csv(os.path.join(saveFolPath, self.folName + '_pos.csv'), encoding='utf-8', sep="\t", index=False)
        emgDf.to_csv(os.path.join(saveFolPath, self.folName + '_emg.csv'), encoding='utf-8', sep="\t", index=False)

        with open(os.path.join(saveFolPath, self.folName + '_fps.txt'), 'w') as file:
            frameRate = str(self.tracker.frameRate)
            file.write(frameRate)

if __name__ == '__main__':
    mainFol = 'MF_0307'
    try:
        os.mkdir(mainFol)
    except FileExistsError as e:
        pass

    if len(sys.argv) == 1:
        name = '__Testing__'
        print(f'Starting hand tracker with savename {name}')
    elif len(sys.argv) == 2:
        name = str(sys.argv[1])
        print(f'Starting hand tracker with savename {name}')
    else:
        raise Exception(f'Wrong number of arguments ({len(sys.argv) - 1})')

    tracker = EMG_hand_recorder(mainFol, name)
    sleep(0.25) # wait an extra quarter second to let the board get up and running?
    tracker.trackAndRecord()

    print('Done.')
