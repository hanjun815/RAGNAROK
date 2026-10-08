# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass

####################################################################################

# Import Modules

import os
import cv2
import numpy as np
import argparse
import csv
import sys

####################################################################################

# Parse command line arguments
if len(sys.argv) != 2:
    print("ERROR! wrong number of arguments")
    print("Usage: python3 convert_to_tum.py [path_to_euroc_folder]")
    sys.exit()

file_to_convert = sys.argv[1]
print("converting data under "+file_to_convert)

out = open(file_to_convert[:-4] + '_tum.txt', 'w')
out.write('# timestamp_s tx ty tz qx qy qz qw\n')
# skip first line i.e. read header first and then iterate over each row od csv as a list
with open(file_to_convert, 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            timestamp = np.float64(row[0])/1E+09
            px = np.float64(row[1])
            py = np.float64(row[2])
            pz = np.float64(row[3])
            qx = np.float64(row[4])
            qy = np.float64(row[5])
            qz = np.float64(row[6])
            qw = np.float64(row[7])
            out.write(str(timestamp))
            out.write(' ')
            out.write(str(px))
            out.write(' ')
            out.write(str(py))
            out.write(' ')
            out.write(str(pz))
            out.write(' ')
            out.write(str(qx))
            out.write(' ')
            out.write(str(qy))
            out.write(' ')
            out.write(str(qz))
            out.write(' ')
            out.write(str(qw))
            out.write('\n')

out.close()


