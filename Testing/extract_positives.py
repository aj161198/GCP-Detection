import os

path = '/home/aman/Desktop/SkyLark Drones/NH-150_Flight/Geotagged-Images/'
files = os.listdir(path)

for file in files:
	if os.path.isfile('/home/aman/Desktop/SkyLark Drones/NH-150_Flight/TruePositive/' + file):
		os.rename('/home/aman/Desktop/SkyLark Drones/NH-150_Flight/Geotagged-Images/' + file, '/home/aman/Desktop/SkyLark Drones/NH-150_Flight/Positive/' + file);
