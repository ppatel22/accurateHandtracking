% Mikey Fernandez
% 11/28/2022
% Plotting tracked Data from hand tracking videos

%%
clc; clear; close all;

%% load data
fileName = 'handTrackingCoordinates_2022-12-08_13:42:46';
data = readtable(['./data/' fileName '.csv']);
frameRate = 30; % 30 FPS video

%% extract for each relevant point
addDim = @(name) [name + "_X", name + "_Y", name + "_Z"];

wrist = table2array(data(:, addDim("WRIST")));

thumb_cmc = table2array(data(:, addDim("THUMB_CMC")));
thumb_mcp = table2array(data(:, addDim("THUMB_MCP")));
thumb_ip = table2array(data(:, addDim("THUMB_IP")));
thumb_tip = table2array(data(:, addDim("THUMB_TIP")));

index_mcp = table2array(data(:, addDim("INDEX_FINGER_MCP")));
index_pip = table2array(data(:, addDim("INDEX_FINGER_PIP")));
index_dip = table2array(data(:, addDim("INDEX_FINGER_DIP")));
index_tip = table2array(data(:, addDim("INDEX_FINGER_TIP")));

middle_mcp = table2array(data(:, addDim("MIDDLE_FINGER_MCP")));
middle_pip = table2array(data(:, addDim("MIDDLE_FINGER_PIP")));
middle_dip = table2array(data(:, addDim("MIDDLE_FINGER_DIP")));
middle_tip = table2array(data(:, addDim("MIDDLE_FINGER_TIP")));

ring_mcp = table2array(data(:, addDim("RING_FINGER_MCP")));
ring_pip = table2array(data(:, addDim("RING_FINGER_PIP")));
ring_dip = table2array(data(:, addDim("RING_FINGER_DIP")));
ring_tip = table2array(data(:, addDim("RING_FINGER_TIP")));

pinky_mcp = table2array(data(:, addDim("PINKY_MCP")));
pinky_pip = table2array(data(:, addDim("PINKY_PIP")));
pinky_dip = table2array(data(:, addDim("PINKY_DIP")));
pinky_tip = table2array(data(:, addDim("PINKY_TIP")));

%% generate the desired vectors and calculate angless
% thumb
thumb1 = thumb_cmc - wrist;
thumb2 = thumb_mcp - thumb_cmc;
thumb3 = thumb_ip - thumb_mcp;
thumb4 = thumb_tip - thumb_ip;
thumbAngle = calcAngles(thumb1, thumb2);
thumbAngle2 = calcAngles(thumb2, thumb3);
thumbAngle3 = calcAngles(thumb3, thumb4);
thumbMean = mean([thumbAngle thumbAngle2 thumbAngle3], 2);

% index
index1 = index_mcp - wrist;
index2 = index_pip - index_mcp;
index3 = index_dip - index_pip;
index4 = index_tip - index_dip;
indexAngle = calcAngles(index1, index2);
indexAngle2 = calcAngles(index2, index3);
indexAngle3 = calcAngles(index3, index4);
indexMean = mean([indexAngle indexAngle2 indexAngle3], 2);

% middle
middle1 = middle_mcp - wrist;
middle2 = middle_pip - middle_mcp;
middle3 = middle_dip - middle_pip;
middle4 = middle_tip - middle_dip;
middleAngle = calcAngles(middle1, middle2);
middleAngle2 = calcAngles(middle2, middle3);
middleAngle3 = calcAngles(middle3, middle4);
middleMean = mean([middleAngle middleAngle2 middleAngle3], 2);

% ring
ring1 = ring_mcp - wrist;
ring2 = ring_pip - ring_mcp;
ring3 = ring_dip - ring_pip;
ring4 = ring_tip - ring_dip;
ringAngle = calcAngles(ring1, ring2);
ringAngle2 = calcAngles(ring2, ring3);
ringAngle3 = calcAngles(ring3, ring4);
ringMean = mean([ringAngle ringAngle2 ringAngle3], 2);

% pinky
pinky1 = pinky_mcp - wrist;
pinky2 = pinky_pip - pinky_mcp;
pinky3 = pinky_dip - pinky_pip;
pinky4 = pinky_tip - pinky_dip;
pinkyAngle = calcAngles(pinky1, pinky2);
pinkyAngle2 = calcAngles(pinky2, pinky3);
pinkyAngle3 = calcAngles(pinky3, pinky4);
pinkyMean = mean([pinkyAngle pinkyAngle2 pinkyAngle3], 2);

% time vector
t = (0:length(thumbAngle) - 1)/frameRate;

%% plot
figure(3); clf
subplot(511)
plot(t, thumbAngle, t, thumbAngle2, t, thumbAngle3, t, thumbMean)
ylabel('Thumb')
subplot(512)
plot(t, indexAngle, t, indexAngle2, t, indexAngle3, t, indexMean)
ylabel('Index')
subplot(513)
plot(t, middleAngle, t, middleAngle2, t, middleAngle3, t, middleMean)
ylabel('Middle')
subplot(514)
plot(t, ringAngle, t, ringAngle2, t, ringAngle3, t, ringMean)
ylabel('Ring')
subplot(515)
plot(t, pinkyAngle, t, pinkyAngle2, t, pinkyAngle3, t, pinkyMean)
ylabel('Pinky')
xlabel('Time (s)')
legend('Rel Wrist', 'Rel MCP', 'Rel PIP', 'Mean')
sgtitle('Possible finger angles')

%% plots
figure(1); clf
hold on
plot(t, thumbAngle)
plot(t, indexAngle)
plot(t, middleAngle)
plot(t, ringAngle)
plot(t, pinkyAngle)
xlabel('Time (s)')
ylabel('Angle (rad)')
title(fileName, 'Interpreter', 'none')
legend('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')

figure(2); clf
hold on
plot(t(1:4:end), downsample(movmean(thumbAngle, 4), 4))
plot(t(1:4:end), downsample(movmean(indexAngle, 4), 4))
plot(t(1:4:end), downsample(movmean(middleAngle, 4), 4))
plot(t(1:4:end), downsample(movmean(ringAngle, 4), 4))
plot(t(1:4:end), downsample(movmean(pinkyAngle, 4), 4))
xlabel('Time (s)')
ylabel('Angle (rad)')
title(fileName, 'Interpreter', 'none')
legend('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')

%% save image
hgexport(gcf, [fileName '.jpg'], hgexport('factorystyle'), 'Format', 'jpeg');

%% functions
function angle = calcAngles(vec1, vec2)
    cosTheta = max(min(dot(vec1, vec2, 2)./(vecnorm(vec1').*vecnorm(vec2'))', 1), -1);
    angle = real(acos(cosTheta));
end