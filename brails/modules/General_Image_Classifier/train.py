# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|                         BRAILS                        |
|                                                       |
| Author: Charles Wang,  UC Berkeley, c_w@berkeley.edu  |
|                                                       |
| Date:    10/15/2020                                   |
*------------------------------------------------------*/
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.inception_v3 import preprocess_input

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Train a classifier')

parser.add_argument('--imgDir',help='The path of a folder containing images in subfolders',type=str,required=True)
parser.add_argument('--modelDir',help='The path of a folder where model will be saved',type=str,required=True)
parser.add_argument('--modelFile',default='classifier.h5',help='The name of the model file (*.h5) to be saved',type=str)
parser.add_argument('--batch_size',default=32, help='Batch size',type=int)
parser.add_argument('--lr1',default=0.0001, help='Learning rate 1',type=float)
parser.add_argument('--lr2',default=0.00001, help='Learning rate 2',type=float)
parser.add_argument('--it1',default=300, help='Iterations 1',type=int)
parser.add_argument('--it2',default=1500, help='Iterations 2',type=int)
parser.add_argument('--dropout',default=0.6, help='Dropout',type=float)
parser.add_argument('--horizontalFlip',default=False, help='Horizontal flip',type=str2bool)
parser.add_argument('--verticalFlip',default=False, help='Vertical flip',type=str2bool)
parser.add_argument('--randomSeed', help='Random seed',type=int)
parser.add_argument('--randomRotation',default=0.0, help='Random rotation',type=float)

parser.add_argument('--plot',default=True, help='Plot figures',type=str2bool)

args = parser.parse_args()

imgDir = args.imgDir
modelDir = args.modelDir
lr1 = args.lr1
lr2 = args.lr2
it1 = args.it1
it2 = args.it2
dropout = args.dropout
horizontalFlip = args.horizontalFlip
verticalFlip = args.verticalFlip
randomseed = args.randomSeed if args.randomSeed else random.randrange(1e8)
randomRotation = args.randomRotation
modelFile = args.modelFile


print('imgDir = {}'.format(imgDir))
print('modelDir = {}'.format(modelDir))
print('lr1 = {}'.format(lr1))
print('lr2 = {}'.format(lr2))
print('it1 = {}'.format(it1))
print('it2 = {}'.format(it2))
print('dropout = {}'.format(dropout))
print('horizontalFlip = {}'.format(horizontalFlip))
print('verticalFlip = {}'.format(verticalFlip))
print('randomseed = {}'.format(randomseed))
print('randomRotation = {}'.format(randomRotation))
print('args.batch_size = {}'.format(args.batch_size))
print('modelFile = {}'.format(modelFile))


## 1. Prepare data

image_size = (256, 256)
batch_size = args.batch_size # 32

print('* First split the data with 8:2.')
train_ds = image_dataset_from_directory(
    imgDir,
    validation_split=0.2,
    subset="training",
    seed=randomseed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)
val_ds = image_dataset_from_directory(
    imgDir,
    validation_split=0.2,
    subset="validation",
    seed=randomseed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)


print("* Then take 20% out of the validation set for final testing.")
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

class_names = train_ds.class_names
print('The names of the classes are: ',class_names)


# Configure the dataset for performance 
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)




## 2. Create model

### 2.1 Load the pre-trained base model

# Load InceptionV3 model pre-trained on imagenet
base_model = tf.keras.applications.InceptionV3(input_shape=image_size + (3,),
                                               include_top=False,
                                               weights='imagenet')
# Freeze the base model
base_model.trainable = False

### 2.2 Add preprocessing layers and a classification head to build the model

# Augmentation
aug_list = []
if horizontalFlip: 
    aug_list.append(tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'))
    print('Horizontal flip applied')
if verticalFlip: 
    aug_list.append(tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'))
    print('Vertical flip applied')
if randomRotation>0.0: 
    aug_list.append(tf.keras.layers.experimental.preprocessing.RandomRotation(randomRotation))
    print('Random rotation = {} applied'.format(randomRotation))
if len(aug_list)>0: data_augmentation = tf.keras.Sequential(aug_list)

# Pre-processing layer
inputs = tf.keras.Input(shape=image_size + (3,))
if len(aug_list)>0: 
    x = data_augmentation(inputs) # augment
    x = preprocess_input(x) 
else: x = preprocess_input(inputs) 


# Then go into the backbone model
x = base_model(x)

# Then go into the classification header
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(dropout)(x) # You can change the dropout rate 
prediction_layer = tf.keras.layers.Dense(len(class_names), activation='softmax')
outputs = prediction_layer(x)

# Put them together
model = tf.keras.Model(inputs, outputs)

### 2.3 Compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr1),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

## 3. Train the model

initial_epochs = it1
history = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds)

# Plot learning curves

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

if args.plot:
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

## 4. Fine tuning

### 4.1 Un-freeze the top layers of the model

# Un-freeze the whole base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 300 # There are a total of 311 layer

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

### 4.2 Compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr2),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#model.summary()

### 4.3 Continue training the model

fine_tune_epochs = it2
total_epochs =  initial_epochs + fine_tune_epochs
history_fine = model.fit(train_ds, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=val_ds)

# Plot learning curves

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

if args.plot:
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

## 5. Evaluate the performance of the model

# Evaluate the overall performance on the test set
loss, accuracy = model.evaluate(test_ds)
print('Test accuracy :', accuracy)

# Save the model in the current folder 
model_path = os.path.join(modelDir, modelFile)
model.save(model_path) 
print('Model saved at ', model_path)
