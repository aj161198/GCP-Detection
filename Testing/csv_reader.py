import csv 
import sys
import re
import numpy as np
import os 

new_f = open("csv_output.txt", 'w') 


file_names = os.listdir("Positives/")

for file_name in file_names:
	name = ""
	f = open(sys.argv[1], 'rb')
	reader = csv.reader(f)
	for row in reader:
		print (file_name)
		name = row[0].replace('Amplus_Bikaner_M1_U1\\', '');
		if (file_name == name):	
			new_f.write(file_name + "\n")
			cordinates = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'\1',row[1])
			cordinates = cordinates.replace(' ', '')
			new_f.write(cordinates + "\n")

f.close()
new_f.close()
