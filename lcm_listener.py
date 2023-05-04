import lcm
import select
from joints1 import jointAngles1
from joints2 import jointAngles2

def my_handler1(channel, data):
    msg = jointAngles1.decode(data)
    print("   timestamp   = %s" % str(msg.timestamp))
    angles = [msg.angle0, msg.angle1, msg.angle2, msg.angle3, msg.angle4, msg.angle5, msg.angle6, msg.angle7]
    print("Angles from first tracker:", angles)

def my_handler2(channel, data):
    msg = jointAngles2.decode(data)
    print("   timestamp   = %s" % str(msg.timestamp))
    angles = [msg.angle0, msg.angle1, msg.angle2, msg.angle3, msg.angle4, msg.angle5, msg.angle6, msg.angle7]
    print("Angles from second tracker:", angles)

lc1 = lcm.LCM()
lc2 = lcm.LCM()
subscription1 = lc1.subscribe("firstTracker", my_handler1)
subscription2 = lc1.subscribe("secondTracker", my_handler2)

# try:
#     while True:
#         lc1.handle()
#         lc2.handle()
# except KeyboardInterrupt:
#     pass


try:
    timeout = .5  # amount of time to wait, in seconds
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
    pass


lc1.unsubscribe(subscription1)
lc2.unsubscribe(subscription2)