from ouster.sdk import client
import numpy as np
import sys
import os
import csv
import rospy
import struct

import rosbag, rospy
import sensor_msgs.msg
from sensor_msgs.msg import Image, Imu, PointCloud2
import cv2
from cv_bridge import CvBridge

"""
PointField[]            
                # This message holds the description of one point entry in the
                # PointCloud2 message format.
                uint8 INT8    = 1
                uint8 UINT8   = 2
                uint8 INT16   = 3
                uint8 UINT16  = 4
                uint8 INT32   = 5
                uint8 UINT32  = 6
                uint8 FLOAT32 = 7
                uint8 FLOAT64 = 8
                
                
Example of a Ouster Point Cloud Msg

header: 
  seq: 0
  stamp: 
    secs: 161
    nsecs: 847092750
  frame_id: "os_sensor"
height: 128
width: 2048
fields: 
  - 
    name: "x"
    offset: 0
    datatype: 7
    count: 1
  - 
    name: "y"
    offset: 4
    datatype: 7
    count: 1
  - 
    name: "z"
    offset: 8
    datatype: 7
    count: 1
  - 
    name: "intensity"
    offset: 12
    datatype: 7
    count: 1
  - 
    name: "t"
    offset: 16
    datatype: 6
    count: 1
  - 
    name: "reflectivity"
    offset: 20
    datatype: 4
    count: 1
  - 
    name: "ring"
    offset: 22
    datatype: 2
    count: 1
  - 
    name: "ambient"
    offset: 23
    datatype: 4
    count: 1
  - 
    name: "range"
    offset: 25
    datatype: 6
    count: 1
is_bigendian: False
point_step: 29
row_step: 59392

https://docs.python.org/3/library/struct.html

=> results in format = 'ffffLHBHLxxxxxxxxxxx' missing pad bytes in the end?

"""

# Ouster sensor setting; for VBR can be 2048x128 or 1024x64
#num_channels = 128
num_channels = 64
#num_hpoints = 2048
num_hpoints = 1024

# make sure that command line argument (filename given)
number_of_bags = len(sys.argv) - 1
if not number_of_bags > 0:
    sys.exit("No bag files provided.")
else:
    bagfiles = sys.argv[1:]
    print("Converting the following bags: ")
    for bag_file in bagfiles:
        print(bag_file)


# Now Iterate bag files and process
print("#########################")
for bag_file in bagfiles:
    print("Processing " + bag_file)

    # Open bag file
    bag_data = rosbag.Bag(bag_file, 'r')
    bridge = CvBridge()

    # Create folder if not existing
    folder_name = bag_file[:-4]
    if not os.path.exists(folder_name):
        print(f"Trying to create folder {folder_name}")
        os.mkdir(folder_name)
        os.mkdir(folder_name + '/lidar0/')
        os.mkdir(folder_name + '/cam0/')
        os.mkdir(folder_name + '/cam0/data/')
        os.mkdir(folder_name + '/cam1/')
        os.mkdir(folder_name + '/cam1/data/')
        os.mkdir(folder_name + '/imu0/')
    else:  # clear folder
        answer = input(f"Are you sure you want to delete '{folder_name}'? (y/N): ").strip().lower()
        if answer == "y":
            os.system('rm -r ' + folder_name)
            print("Folder deleted.")
        else:
            print("Canceled.")
            sys.exit()
        print(f"Trying to renew folder {folder_name}")
        os.mkdir(folder_name + '/lidar0/')
        os.mkdir(folder_name + '/cam0/')
        os.mkdir(folder_name + '/cam0/data/')
        os.mkdir(folder_name + '/cam1/')
        os.mkdir(folder_name + '/cam1/data/')
        os.mkdir(folder_name + '/imu0/')

    # lidar byte format
    lidar_msg_format = 'ffffIHBHI'
    lidar_msg_format = 'ffffIHBHf'
    lidar_msg_format = 'fffxxxxxxxxxxxxxxxxx'
    lidar_msg_format = '<fff f I H B H I'

    # Prepare CSV Files
    cam0_csvFile = open(folder_name + "/cam0/data.csv", 'w')
    cam0_csvWriter = csv.writer(cam0_csvFile)
    cam0_csvWriter.writerow(['#timestamp [ns]', 'filename'])

    cam1_csvFile = open(folder_name + "/cam1/data.csv", 'w')
    cam1_csvWriter = csv.writer(cam1_csvFile)
    cam1_csvWriter.writerow(['#timestamp [ns]', 'filename'])

    imu_csvFile = open(folder_name + "/imu0/data.csv", 'w')
    imu_csvWriter = csv.writer(imu_csvFile)
    imu_csvWriter.writerow(
        ['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 'a_RS_S_x [m s^-2]',
         'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]'])

    lidar_csvFile = open(folder_name + "/lidar0/data.csv", 'w')
    lidar_csvWriter = csv.writer(lidar_csvFile)
    lidar_csvWriter.writerow(
        ['#timestamp [ns]', 'x', 'y', 'z', 'intensity', 'ring'])

    for topic, msg, t in bag_data:
        # Cam0
        if topic == '/camera_left/image_raw':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imwrite(folder_name + '/cam0/data/' + str(timestamp) + ".png", cv_img)
            cam0_csvWriter.writerow([str(timestamp), str(timestamp) + ".png"])

        # Cam1
        if topic == '/camera_right/image_raw':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imwrite(folder_name + '/cam1/data/' + str(timestamp) + ".png", cv_img)
            cam1_csvWriter.writerow([str(timestamp), str(timestamp) + ".png"])

        # IMU
        if topic == '/imu/data':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            acc_x = msg.linear_acceleration.x
            acc_y = msg.linear_acceleration.y
            acc_z = msg.linear_acceleration.z
            gyr_x = msg.angular_velocity.x
            gyr_y = msg.angular_velocity.y
            gyr_z = msg.angular_velocity.z
            imu_csvWriter.writerow([str(timestamp), gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z])

        # Lidar
        if topic == '/ouster/points':     
            print(f"Lidar message t={t}")
            print(f"Lidar message header t={msg.header.stamp}")

            # Lidar
            header_timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            # access uint8_t[] data array and process according to PointField[] specification
            test_iter = struct.iter_unpack(lidar_msg_format, msg.data)
            
            # Buffer measurements per msg to sort for timestamps
            lidar_ts = np.zeros([num_channels*num_hpoints])
            lidar_x = np.zeros([num_channels*num_hpoints])
            lidar_y = np.zeros([num_channels*num_hpoints])
            lidar_z = np.zeros([num_channels*num_hpoints])
            lidar_intensity = np.zeros([num_channels*num_hpoints])
            lidar_ring = np.zeros([num_channels*num_hpoints])


            count = 0
            row_cnt = 0
            for packet in test_iter:
                # packet will be a tuple (x,y,z, intensity, timestamp, reflectivity, ambient, range)
                
                delta_timestamp = rospy.Time.from_seconds(packet[4]*1e-09)
                total_timestamp = rospy.Time(header_timestamp.secs, header_timestamp.nsecs+packet[4])
                lidar_ts[count] = total_timestamp.to_nsec()
                
                x = packet[0]
                y = packet[1]
                z = packet[2]
                lidar_x[count] = x
                lidar_y[count] = y
                lidar_z[count] = z
                
                intensity = packet[3]
                lidar_intensity[count] = intensity
                ring = packet[6]
                lidar_ring[count] = ring
                
                if packet[4] == 0:
                    row_cnt+=1
                count+=1
                
                
            # message fully processed
            if row_cnt == num_channels:
                sidx = np.argsort(lidar_ts)
                slidar_ts = lidar_ts[sidx]
                slidar_x = lidar_x[sidx]
                slidar_y = lidar_y[sidx]
                slidar_z = lidar_z[sidx]
                slidar_intensity = lidar_intensity[sidx]
                slidar_ring = lidar_ring[sidx]

                for i in range(0,count):
                    x = slidar_x[i]
                    y = slidar_y[i]
                    z = slidar_z[i]
                    if x !=0 and y!=0 and z !=0:
                        ts = slidar_ts[i]
                        intensity = slidar_intensity[i]
                        ring = slidar_ring[i]
                        lidar_csvWriter.writerow([str(ts), x, y, z, intensity, ring])

            print(f"Detected {count} lidar points in Pointcloud")

    # Clsoe Csv files
    cam0_csvFile.close()
    cam1_csvFile.close()
    imu_csvFile.close()
    lidar_csvFile.close()
    print("Finished Processing " + bag_file)
    # Close the bag
    bag_data.close()

print("Finished Processing all bag files")

