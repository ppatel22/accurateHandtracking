import lcm
import select
import pandas as pd
from joints1 import jointAngles1
from joints2 import jointAngles2

# NOTE: This file takes in the angles from each tracker and saves them to two different
# pandas DataFrames. Post-processing can be done on the data after it has been saved as a .csv
# Columns named angle1.X relate to the first tracker and angle2.X relates to the second.

firstData = pd.DataFrame(
    columns=[
        "angle1.0",
        "angle1.1",
        "angle1.2",
        "angle1.3",
        "angle1.4",
        "angle1.5",
        "angle1.6",
        "angle1.7",
    ]
)

secondData = pd.DataFrame(
    columns=[
        "angle2.0",
        "angle2.1",
        "angle2.2",
        "angle2.3",
        "angle2.4",
        "angle2.5",
        "angle2.6",
        "angle2.7",
    ]
)
i = 0
j = 0


def my_handler1(channel, data):
    msg = jointAngles1.decode(data)
    print("   timestamp   = %s" % str(msg.timestamp))
    angles = [
        msg.angle0,
        msg.angle1,
        msg.angle2,
        msg.angle3,
        msg.angle4,
        msg.angle5,
        msg.angle6,
        msg.angle7,
    ]
    print("Angles from first tracker:", angles)
    firstData.loc[i] = angles
    i += 1


def my_handler2(channel, data):
    msg = jointAngles2.decode(data)
    print("   timestamp   = %s" % str(msg.timestamp))
    angles = [
        msg.angle0,
        msg.angle1,
        msg.angle2,
        msg.angle3,
        msg.angle4,
        msg.angle5,
        msg.angle6,
        msg.angle7,
    ]
    print("Angles from second tracker:", angles)
    secondData.loc[j] = angles
    j += 1


lc1 = lcm.LCM()
lc2 = lcm.LCM()
subscription1 = lc1.subscribe("firstTracker", my_handler1)
subscription2 = lc1.subscribe("secondTracker", my_handler2)

try:
    timeout = 0.5  # amount of time to wait, in seconds
    while True:
        rfds, wfds, efds = select.select([lc1.fileno()], [], [], timeout)
        if rfds:
            lc1.handle()
        else:
            print("Waiting for message from first tracker...")
        rfds, wfds, efds = select.select([lc2.fileno()], [], [], timeout)
        if rfds:
            lc2.handle()
        else:
            print("Waiting for message from second tracker...")
except KeyboardInterrupt:
    merged = pd.concat([firstData, secondData], ignore_index=True)
    merged.to_save("data.csv", index=False)
    pass


lc1.unsubscribe(subscription1)
lc2.unsubscribe(subscription2)
