"""
It is advised that GCP's used should not be placed near white regions or should not be occluded by external factors
The images should be taken in day-light scenarios
"""

from gcp_detect_module import bbox_detector, bbox_verification
import csv
import numpy as np
import os

# path to the folder to be tested
path = "/home/aman/Desktop/SkyLark Drones/Amplus_Bikaner_M1_U1/Positives"
file_names = os.listdir(path)

# path to the csv file
f = open("/home/aman/Desktop/SkyLark Drones/Amplus_Bikaner_M1_U1/csv_output.txt", 'r')

# reads csv file
reader = csv.reader(f)
content = ""

# Switch to change the control between if and else
count = 1

# Looping through the csv-file to detect and verify gcps.
for row in reader:
    if count == 1:
        # Prints the filename and gives all the data of bboxs
        print row[0],
        content = content + row[0] + ","
        row = os.path.join(path, row[0])
        img_file_name = row
        answer = detect
    count = count * -1

# Writes all details into the file
f1 = open("/home/aman/CLEANMAX_lower.csv", 'w')
f1.write(content)
f1.close()
