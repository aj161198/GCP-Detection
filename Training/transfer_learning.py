#This file loads in the mnist model, and tries to do deep-learning on it, and does the binary classification in 100 epochs


from keras.layers import Input
from keras.models import Model
import numpy as np
import pandas as pd
from load_data import load_data

# model.save('mnist.h5')
inputs = Input([28,28,1])

mnist = Model(inputs = model.layers[0].input,  outputs= model.get_layer('dense1').output)

mnist_output = mnist(inputs)
# dropout = Dropout(0.5)(mnist_output)

X_train, Y_train, X_validation , Y_validation = load_data('pos', 'Negs', val_split= 0.2)
print(X_train.shape, Y_train.shape, X_validation.shape , Y_validation.shape)

dense = Dense(2, activation = 'softmax')(mnist_output)

main_model = Model(inputs = inputs , outputs = dense, name = 'aman_proj')
main_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
main_model.fit(X_train, Y_train, validation_data = [X_validation, Y_validation], shuffle = True, epochs = 100)
main_model.save('main_model1.h5')
