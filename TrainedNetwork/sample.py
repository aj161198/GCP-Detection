#Calculates the false-positives in the directory

import cv2
import numpy as np
import os
from classifier import classifier
falsepos = 0
obj = classifier()
file_names = os.listdir("/home/aman/Desktop/stats/Negs/")
for file_name in file_names:
	img = cv2.imread("/home/aman/Desktop/stats/Negs/" + file_name, 0)
	img = cv2.resize(img, (28, 28))
	ans =  obj.classify(img)
	print ans
	if (np.argmax(ans[0]) == 1):
		falsepos = falsepos + 1
		print falsepos
print falsepos
#4,10
