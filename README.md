Automatic GCP Detection

To carry on various photogrammetry operation, there is a need of accquiring some sort of correspondences between Image and World Cordinate Systems. GCP's help in providing this correspondences, in different scenarios, be it generating ortho-mosaics, DSM or DEMS, etc. But getting this information about GCP's is a hectic task; In world-frame it is obtained very accurately using some specialized GPS units. And in images it is done by extracting the image cordinates of GCP's, manually, which is very much human-intensive. This module helps in automating the latter half of acquiring correspondences, using the amalgamation of Computer-Vision and Neural-Networks

Getting Started

1. git clone the repositiory.
2. Fetch all the dependencies listed in the end.
3. Place the main.py file in the trained network
4. Copy images from FalseNegatives and True  
5. Run the csv_reader.py passing GCP_location.csv as argument
 Run the python code.

Author

Aman Jain
aj161198@gmail.com
