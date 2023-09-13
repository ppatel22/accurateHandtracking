# accurateHandtracking
Utilizes Google's Mediapipe tools and Pandas to improve accuracy of finger joint angle readings from two cameras simultaneously.

Background:
Current gen neuroposthetic hands (meaning they read some form of neural signals to determine movements) are restricted to discrete, motions, i.e. pointing, closed fist, opened fist, etc. The human hand does not work in a discrete manner, but rather allows for continous motion between different hand "poses". In efforts to improve the current design, the Biomechatronics lab at the MIT Media Lab is creating a more robust mapping between EEG signals from the forearm to intended movement in the hands. EEG signals come from muscles, and since the muscles and nerves that control hands are in the forearms, we are able to predict hand movement on some amputees even though they no longer possess the actual limb. Note that this will only work on the amputees that have not lost the relevant nerves.

Usage:
When collecting data to create the mapping between nerves firing and intended motion, we track the motion of the existing hand on the patient/subject. Later, the continuous joint angles can be used as ground truth for an algorithm that predicts intended motion of the hand.

Prerequisites:
You must be on the latest version of Python. Please ensure that the following packages are installed before running this code: opencv2, mediapipe, numpy, and pandas. You can paste the line below into your terminal to install the necessary packages.

```pip install opencv-python numpy pandas mediapipe```

To learn more about Google's MediaPipe tools, please visit their [site](https://developers.google.com/mediapipe/solutions/examples).

Running the code:
Once all the preperations have been completed, attach at least one external webcam to your computer (both are required if your computer does not come with a built-in webcam). Once in the correct directory, open up three different terminal tabs. Input the following lines of code into each terminal:

1) ```python trackerL.py x```
2) ```python trackerR.py y```

Note that "x" and "y" will take on an integer value that tells the program which video inputs to use. The default webcam is typically assigned to 0, while the second webcam is usually 1.