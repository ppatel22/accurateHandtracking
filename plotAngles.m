% Mikey Fernandez
% 12/05/2022
% Plotting tracked Data from hand tracking videos

%%
clc; clear; close all;

%% load data
fileName = 'Slo-Mo Tracking';
data = readtable(['./data/' fileName '.csv']);
frameRate = 240; % 30 FPS video

%% extract for each relevant point
% jointNames = ["thumbPPos", "thumbYPos", "indexPos", "mrpPos", "wristRot", "wristFlex", "humPos", "elbowPos"];
jointNames = ["thumbPPos", "thumbYPos", "indexPos", "mrpPos"];

jointPos = table2array(data(:, jointNames));
timeArray = (0:length(jointPos) - 1)/frameRate;

%% Analyze data
figure
plot(timeArray, jointPos)
xlabel('Time (s)')
ylabel('Angle (deg)')
legend(jointNames)