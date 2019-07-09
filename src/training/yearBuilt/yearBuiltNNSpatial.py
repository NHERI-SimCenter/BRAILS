# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
| This script is used to train a neural net to predict  |
| SAM parameters given BIM features.                    |
|                                                       |
| Author: Charles Wang,  UC Berkeley c_w@berkeley.edu   |
|                                                       |
| Date:   01/09/2019                                    |
*------------------------------------------------------*/
"""

from __future__ import absolute_import, division, print_function
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

# fix random seed for reproducibility
tf.set_random_seed(1234)


# define paths
dataset_path = '../data/data_x.csv'
label_path = '../data/data_y.csv'



# load data
column_names = ['x', 'y', 'd1', 'v1', 'd2', 'v2', 'd3', 'v3', 'd4', 'v4', 'd5', 'v5'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

# load labels:   
column_names= ['label']
raw_labelset = pd.read_csv(label_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)
labelset = raw_labelset[['label']].copy()
print(labelset.tail())


# merge label into dataset
dataset = pd.concat([dataset, labelset], axis=1, join='inner')
print(dataset.tail())


# Split data into train and test
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# statistics
train_stats = train_dataset.describe()
train_stats.pop("label")
train_stats = train_stats.transpose()
print(train_stats)
store = pd.HDFStore('../data/data.h5')
store['stats'] =  train_stats


# Split features from labels
train_labels = train_dataset[['label']].copy()
train_dataset = train_dataset.drop(['label'], axis=1)
test_labels = test_dataset[['label']].copy()
test_dataset = test_dataset.drop(['label'], axis=1)



# Normalize data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)




# Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])
  #optimizer = tf.train.RMSPropOptimizer(0.001)
  optimizer = tf.train.AdamOptimizer(1e-4)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
  return model



# Training
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: 
      print(epoch)
    print('.', end='')

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error ')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  #plt.ylim([0,1])
  '''
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Ap^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  #plt.ylim([0,1])
  '''
  #plt.show()

EPOCHS = 5000

'''
# no early stop
model = build_model()
model.summary()
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\n')
print(hist.tail())
plot_history(history)
'''


model = build_model()
model.summary()
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\n')
print(hist.tail())
plot_history(history)
plt.savefig('../data/NN_ContinuumWall_TrainingLoss_V1.png')


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))



# serialize model to JSON
model_json = model.to_json()
with open("../data/NNModel_ContinuumWall_V1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../data/NNModel_ContinuumWall_V1.h5")
print("Saved model to disk")



# Predict
test_predictions = model.predict(normed_test_data).flatten()



plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
trueValues = test_labels['label']
#predictValues = test_predictions[0::5]
predictValues = test_predictions
plt.scatter(trueValues, predictValues)
plt.xlabel('True Values [label]')
plt.ylabel('Predictions [label]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.subplot(1,2,2)
error = predictValues - trueValues
print('errors:  ')
print(train_dataset)
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [label]")
_ = plt.ylabel("Count")


plt.savefig('../data/NN_ContinuumWall_Predictions_V1.png')
plt.show()
