#!/usr/bin/python
import os
import rospy
import numpy as np
from sloth import sloth
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model_path = os.path.abspath(__file__ + "/../../model/LSTMnet.h5")
c = [3, 3, 3, 3, 3, 3]
tau = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
S = sloth(model_path, 10, 6, 3,0.2,tau,c)

global data_stream 
data_stream =  np.empty((1,100,3))
data_stream[:] = np.nan

global gesture_instance
gesture_instance = np.empty((1,100,3))
gesture_instance[:] = np.nan

global last 
last = 0

def imu_callback(data):
    global last
    global data_stream
    global gesture_instance

    if long(data.header.frame_id) - last > 90:
        #print data.header.frame_id
        gesture_instance = np.roll(gesture_instance,99,1)
        gesture_instance[:,-1,:] = np.nan
        data_stream = np.roll(data_stream,99,1)
        data_stream[:,-1,0] = data.linear_acceleration.x
        data_stream[:,-1,1] = data.linear_acceleration.y
        data_stream[:,-1,2] = data.linear_acceleration.z
        
        last = long(data.header.frame_id)
        S.window_update(data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z)
        S.classify()
        S.detect()

def main():
    global data_stream
    global gesture_instance

    rospy.init_node('sloth', anonymous=True)
    rospy.Subscriber('/imu_left_hand', Imu, imu_callback, queue_size=1)
    r = rospy.Rate(10)

    plt.ion()

    gridsize = (3, 2)
    fig = plt.figure(figsize=(40, 16))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=3)
    ax2 = plt.subplot2grid(gridsize, (0, 1))
    ax3 = plt.subplot2grid(gridsize, (1, 1))
    ax4 = plt.subplot2grid(gridsize, (2, 1))

    ax1.set_xticks([])
    ax1.set_yticks([])

    ticks_label = ["", "0", "2", "4", "6", "8", "10", ""]

    images_path = os.path.abspath(__file__ + "/../../gestures_images/")
    
    while not rospy.is_shutdown():
        #S.display()
        gesture = S.get_gesures()
        for ges in gesture:
            gesture_instance[:,-1,:] = data_stream[:,-1,:]
            img=mpimg.imread(images_path + "/" + str(ges) + ".jpg")
            ax1.imshow(img)

        ax2.clear()
        ax3.clear()
        ax4.clear()

        ax2.set_ylim([-14,14])
        ax3.set_ylim([-14,14])
        ax4.set_ylim([-14,14])

        ax2.set_xlim([-20,120])
        ax3.set_xlim([-20,120])
        ax4.set_xlim([-20,120])

        ax2.set_ylabel("Linear Acc X (m/s^2)")
        ax3.set_ylabel("Linear Acc Y (m/s^2)")
        ax4.set_ylabel("Linear Acc Z (m/s^2)")

        ax2.set_xlabel("Time (s)")
        ax3.set_xlabel("Time (s)")
        ax4.set_xlabel("Time (s)")

        ax2.set_xticklabels(ticks_label)
        ax3.set_xticklabels(ticks_label)
        ax4.set_xticklabels(ticks_label)

        ax2.scatter(range(0,100), gesture_instance[0,:,0],s=100 ,c="g")
        ax3.scatter(range(0,100), gesture_instance[0,:,1],s=100, c="g")
        ax4.scatter(range(0,100), gesture_instance[0,:,2],s=100, c="g")

        ax2.plot(range(0,100),data_stream[0,:,0])
        ax3.plot(range(0,100),data_stream[0,:,1])
        ax4.plot(range(0,100),data_stream[0,:,2])
        fig.canvas.flush_events()
        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
