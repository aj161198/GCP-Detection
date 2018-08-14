from keras.models import load_model
import numpy as np


class classifier(object):
    """
    argument:
    image : image file to be classified
    image shape must be 28* 28 and should have a single channel
    make sure that model.h5 file is in the same directory as this module
    class 1 : positive ( marker )
    class 2 : negative ( non marker )
    returns probability of class 1 and 2 in form of a numpy array
    """

    def __init__(self):
        super(classifier, self).__init__()
        self.prediction = load_model('model.h5')

    def classify(self, img):
        self.img = img
        img = np.array(img)
        return self.prediction.predict(img.reshape([1, 28, 28, 1]))


if '__name__' == '__main__':
    print('Error : only to be used as a module')
