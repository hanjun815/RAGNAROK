
import numpy as np
import cv2
import csv
import os
import sys
import struct

import rosbag, rospy
import sensor_msgs.msg
from sensor_msgs.msg import Image, Imu, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


"""
Script to convert rosbags from HILTI Challenge Datasets to Leica(ASL / EuRoC based) format
https://hilti-challenge.com/dataset-2022.html
https://github.com/Hilti-Research/Hilti-SLAM-Challenge-2022

HILTI Bags contain following topics:
- Camera topics (sensor_msgs/Image)
    /alphasense/cam0/image_raw
    /alphasense/cam1/image_raw
    /alphasense/cam2/image_raw
    /alphasense/cam3/image_raw
    /alphasense/cam4/image_raw
- IMU topic (sensor_msgs/Imu)
    /alphasense/imu
- LiDAR topic
    /hesai/pandar (sensor_msgs/PointCloud2)
    
    The Hesai ros driver stores the timestamp in this struct. What happens is the sensor_msgs/PointCloud2 Message has a 
    "data" member in byte and it stores the PointXYZIT defined time, xyz, etc. The "field" member describes what type of 
    info is in "data". In a programme, one would convert the PointCloud2 msg into PointXYZIT msg to access all the 
    element pandar records.
    
    
    Point Cloud 2 msg (http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html)
    
        height: 1 (=> unordered point cloud)
        width: e.g. 57456 (depending on frame)
        point_step: 48 (48 bytes per point)
        msg.row_step: bytes per row, in this case = point_step * width
                # This message holds a collection of N-dimensional points, which may
                # contain additional information such as normals, intensity, etc. The
                # point data is stored as a binary blob, its layout described by the
                # contents of the "fields" array.
                
                # The point cloud data may be organized 2d (image-like) or 1d
                # (unordered). Point clouds organized as 2d images may be produced by
                # camera depth sensors such as stereo or time-of-flight.
                
                # Time of sensor data acquisition, and the coordinate frame ID (for 3d
                # points).
                Header header
                
                # 2D structure of the point cloud. If the cloud is unordered, height is
                # 1 and width is the length of the point cloud.
                uint32 height
                uint32 width
                
                # Describes the channels and their layout in the binary data blob.
                PointField[] fields
                
                bool    is_bigendian # Is this data bigendian?
                uint32  point_step   # Length of a point in bytes
                uint32  row_step     # Length of a row in bytes
                uint8[] data         # Actual point data, size is (row_step*height)
                
                bool is_dense        # True if there are no invalid points
                
                
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
                
                string name      # Name of field
                uint32 offset    # Offset from start of point struct
                uint8  datatype  # Datatype enumeration, see above
                uint32 count     # How many elements in the field
                
                
        PointField[0]
            name: "x"
            offset: 0
            datatype: 7
            count: 1
        PointField[1]
            name: "y"
            offset: 4
            datatype: 7
            count: 1
        PointField[2]
            name: "z"
            offset: 8
            datatype: 7
            count: 1
        PointField[3]
            name: "intensity"
            offset: 16
            datatype: 7
            count: 1
        PointField[4]
            name: "timestamp"
            offset: 24
            datatype: 8
            count: 1
        PointField[5]
            name: "ring"
            offset: 32
            datatype: 4
            count: 1

    
    
    => results in format = 'fffxxxxfxxxxdHxxxxxxxxxxxxxx'
"""


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
        os.mkdir(folder_name)
    else:  # clear folder
        os.system('rm -rf ' + folder_name)
        os.mkdir(folder_name)

    # Setup folders for images
    os.mkdir(folder_name + '/cam0/')
    os.mkdir(folder_name + '/cam0/data/')
    os.mkdir(folder_name + '/cam1/')
    os.mkdir(folder_name + '/cam1/data/')
    os.mkdir(folder_name + '/cam2/')
    os.mkdir(folder_name + '/cam2/data/')
    os.mkdir(folder_name + '/cam3/')
    os.mkdir(folder_name + '/cam3/data/')
    os.mkdir(folder_name + '/cam4/')
    os.mkdir(folder_name + '/cam4/data/')
    # Setup folder for imu
    os.mkdir(folder_name + '/imu0/')
    # Setup folder for lidar
    os.mkdir(folder_name + '/lidar0/')
    # lidar byte format
    lidar_msg_format = 'fffxxxxfxxxxdHxxxxxxxxxxxxxx'

    # Prepare CSV Files
    cam0_csvFile = open(folder_name + "/cam0/data.csv", 'w')
    cam0_csvWriter = csv.writer(cam0_csvFile)
    cam0_csvWriter.writerow(['#timestamp [ns]', 'filename'])

    cam1_csvFile = open(folder_name + "/cam1/data.csv", 'w')
    cam1_csvWriter = csv.writer(cam1_csvFile)
    cam1_csvWriter.writerow(['#timestamp [ns]', 'filename'])

    cam2_csvFile = open(folder_name + "/cam2/data.csv", 'w')
    cam2_csvWriter = csv.writer(cam2_csvFile)
    cam2_csvWriter.writerow(['#timestamp [ns]', 'filename'])

    cam3_csvFile = open(folder_name + "/cam3/data.csv", 'w')
    cam3_csvWriter = csv.writer(cam3_csvFile)
    cam3_csvWriter.writerow(['#timestamp [ns]', 'filename'])

    cam4_csvFile = open(folder_name + "/cam4/data.csv", 'w')
    cam4_csvWriter = csv.writer(cam4_csvFile)
    cam4_csvWriter.writerow(['#timestamp [ns]', 'filename'])

    imu_csvFile = open(folder_name + "/imu0/data.csv", 'w')
    imu_csvWriter = csv.writer(imu_csvFile)
    imu_csvWriter.writerow(
        ['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 'a_RS_S_x [m s^-2]',
         'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]'])

    lidar_csvFile = open(folder_name + "/lidar0/data.csv", 'w')
    lidar_csvWriter = csv.writer(lidar_csvFile)
    lidar_csvWriter.writerow(
        ['#timestamp [ns]', 'x', 'y', 'z', 'Intensity', 'ring'])

    for topic, msg, t in bag_data:

        # Cam0
        if topic == '/alphasense/cam0/image_raw':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imwrite(folder_name + '/cam0/data/' + str(timestamp) + ".png", cv_img)
            cam0_csvWriter.writerow([str(timestamp), str(timestamp) + ".png"])

        # Cam1
        if topic == '/alphasense/cam1/image_raw':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imwrite(folder_name + '/cam1/data/' + str(timestamp) + ".png", cv_img)
            cam1_csvWriter.writerow([str(timestamp), str(timestamp) + ".png"])

        # Cam2
        if topic == '/alphasense/cam2/image_raw':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imwrite(folder_name + '/cam2/data/' + str(timestamp) + ".png", cv_img)
            cam2_csvWriter.writerow([str(timestamp), str(timestamp) + ".png"])

        # Cam3
        if topic == '/alphasense/cam3/image_raw':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imwrite(folder_name + '/cam3/data/' + str(timestamp) + ".png", cv_img)
            cam3_csvWriter.writerow([str(timestamp), str(timestamp) + ".png"])

        # Cam4
        if topic == '/alphasense/cam4/image_raw':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imwrite(folder_name + '/cam4/data/' + str(timestamp) + ".png", cv_img)
            cam4_csvWriter.writerow([str(timestamp), str(timestamp) + ".png"])

        # IMU
        if topic == '/alphasense/imu':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            acc_x = msg.linear_acceleration.x
            acc_y = msg.linear_acceleration.y
            acc_z = msg.linear_acceleration.z
            gyr_x = msg.angular_velocity.x
            gyr_y = msg.angular_velocity.y
            gyr_z = msg.angular_velocity.z
            imu_csvWriter.writerow([str(timestamp), gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z])

        # Lidar
        if topic == '/hesai/pandar':
            timestamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            # access uint8_t[] data array and process according to PointField[] specification
            test_iter = struct.iter_unpack(lidar_msg_format, msg.data)

            count = 0
            for packet in test_iter:
                # packet will be a tuple (x,y,z, intensity, timestamp, ring)
                count += 1
                x = packet[0]
                y = packet[1]
                z = packet[2]
                intensity = packet[3]
                timestamp = rospy.Time.from_seconds(packet[4])
                ring = packet[5]
                lidar_csvWriter.writerow([str(timestamp), x, y, z, intensity, ring])

            print(f"Detected {count} lidar points in Pointcloud")

    # Clsoe Csv files
    cam0_csvFile.close()
    cam1_csvFile.close()
    cam2_csvFile.close()
    cam3_csvFile.close()
    cam4_csvFile.close()
    imu_csvFile.close()
    lidar_csvFile.close()
    print("Finished Processing " + bag_file)

print("Finished Processing all bag files")

