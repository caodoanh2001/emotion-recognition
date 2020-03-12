import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Dense, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.models import load_model

model = load_model('model/trainedModel_cnn5.h5')
model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

test_data = pd.read_csv("test/pritest.csv")
test_data['pixels'] = test_data['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
test_pix = np.vstack(test_data['pixels'].values)
test_ind = test_data["emotion"].values
test_pix = test_pix.reshape(-1, 48, 48, 1)
prediction = model.predict(test_pix)
class_labels =  prediction.argmax(axis=1)
true = 0
for cond in (class_labels == test_ind):
  if (cond):
    true = true+1
acc = (true/len(prediction))*100
print('Accuracy: ',acc)