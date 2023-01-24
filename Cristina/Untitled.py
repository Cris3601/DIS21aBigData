#!/usr/bin/env python
# coding: utf-8

# In[105]:


# import necessary packages
import os
import shutil

import numpy
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tensorflow.keras.__version__


# ## Prepare Data

# In[110]:


# directorie structure
test_benign = 'skincancer/test/benign'
test_malignant = 'skincancer/test/malignant'

train_benign = 'skincancer/train/benign'
train_malignant = 'skincancer/train/malignant'


# In[97]:


# Erstellen von Listen mit Dateipfaden für jeden Ordner
test_benign_paths = tf.data.Dataset.list_files(test_benign + '/*.jpg')
test_malignant_paths = tf.data.Dataset.list_files(test_malignant + '/*.jpg')
train_benign_paths = tf.data.Dataset.list_files(train_benign + '/*.jpg')
train_malignant_paths = tf.data.Dataset.list_files(train_malignant + '/*.jpg')


# In[98]:


# write function to read and preproces images
def read_and_preprocess_image(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.image.per_image_standardization(image)
    return image


# In[99]:


# Use the map method to read and preprocess the images
test_benign_ds = test_benign_paths.map(read_and_preprocess_image)
test_malignant_ds = test_malignant_paths.map(read_and_preprocess_image)
train_benign_ds = train_benign_paths.map(read_and_preprocess_image)
train_malignant_ds = train_malignant_paths.map(read_and_preprocess_image)


# ## Define the model

# In[107]:


# Instantiate a VGG16 model with pre-trained weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[108]:


# Instantiate a ResNet50 model with pre-trained weights
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[109]:


# Create a convolutional neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# ## Train the model

# In[77]:


# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[78]:


# Train the model
history = model.fit(train_benign_ds, epochs=10, validation_data=test_benign_ds)


# ## Evaluate the model

# In[79]:


# Evaluate the model on the test dataset
test_ds = test_ds.map(process_path)
test_loss, test_acc = model.evaluate(test_ds)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)


# In[100]:


# Save the model
model.save('skin_cancer_classifier.h5')


# In[ ]:




