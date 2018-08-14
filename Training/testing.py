"""This is a sample file for testing, showing how to load model, and get predictions"""

from keras.models import load_model
import cv2
import numpy as np

prediction = load_model('main_model.h5')

def load_test(path):
  x = np.empty([len(os.listdir(path)), 28,28,1])
  count = 0 
  for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))[:,:,0]
    img = img.reshape([28,28,1])
    x[count] = img
    count += 1
    if count == len(os.listdir(path)): break
  return x 

x = load_test('Test')
print('done loading')

prob = prediction.predict(x) 
print(prob)

df = pd.DataFrame(prob, columns = ['positive', 'negative'])
df['filenames'] = os.listdir('Test')
df.to_csv('predictions_on_test.csv')
