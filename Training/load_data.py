#data_load
#This file tries to load_data from the paths given, and fragments it into training-data and testing-data. 

import os 
import cv2
from tqdm import tqdm


def load_data(pos_path, neg_path, val_split):
  length = len(os.listdir(pos_path)) + len(os.listdir(neg_path))
  images  = np.empty([length, 28,28])
  labels = np.empty ([length,2])
  count = 0
  for img in (os.listdir(pos_path)):
    image = cv2.imread(os.path.join(pos_path, img))[:,:,0]
    images[count] = image
    labels[count][0] = 1
    labels[count][1] = 0
    count += 1
    if count / 50 == 0 : print(f'done : {count}')
  print('positives done!')
  for img in (os.listdir(neg_path)):
    image = cv2.imread(os.path.join(neg_path, img))[:,:,0]
    images[count] = image
    labels[count][1] = 1
    labels[count][0] = 0
    count += 1
    if count == length:
      print('negatives done!')
      break
  
  #loop for randomizing
  rimages  = np.empty([length, 28,28])
  rlabels = np.empty ([length,2])
  count = 0 
  for i in np.random.permutation(length):
    rimages[i] = images[count]
    rlabels[i]= labels[count]
    count += 1
  print('randomized')
  split = np.int_((1- val_split)*length)
  rimages = np.reshape(rimages, [length, 28, 28, 1])
  X_train = rimages[:split] 
  X_test = rimages[split:]
  Y_train = rlabels[:split]
  Y_test = rlabels[split:]
  return X_train, Y_train, X_test , Y_test
