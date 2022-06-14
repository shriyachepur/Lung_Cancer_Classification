#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
import horovod.tensorflow.keras as hvd


# In[2]:


os.listdir("Data")


# In[3]:


import argparse
import time
import sys


# In[4]:


from tqdm import tqdm
import cv2


# In[5]:


hvd.init()


# In[7]:


parser = argparse.ArgumentParser()
parser.add_argument('-- epochs', type=int, default=5)
parser.add_argument(' -- batch_size', type=int, default=256)


# In[9]:


args = parser.parse_args()


# In[13]:


gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tensorflow.config.experimental.set_visible_devices(
    gpus[hvd.local_rank()], 'GPU')


# In[14]:


train_data_path = "Data/train"


# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[16]:


from tensorflow.python.keras.preprocessing import image_dataset


# In[17]:


train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=5, width_shift_range=0.2,height_shift_range=0.2,
        shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')


# In[18]:


train_dataset  = train_datagen.flow_from_directory(directory = train_data_path,
                                                   target_size = (224,224),
                                                   class_mode = 'categorical',
                                                   batch_size = 64)


# In[11]:


#i=cv2.imread('/content/gdrive/MyDrive/Projects/Chest CT-Scan/Data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000000 (6).png')


# In[19]:


#cv2_imshow(i)


# In[20]:


from tensorflow.keras.layers import Input,Dense, MaxPool2D, Conv2D, Flatten


# In[21]:


from tensorflow.keras import Sequential, Input


# In[33]:


callbacks =hvd.callbacks.BroadcastGlobalVariablesCallback(0)


# In[38]:


import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet101V2


# In[40]:


model = ResNet101V2(weights="imagenet", include_top=False,input_shape = (224,224,3))
for layer in model.layers:
    layer.trainable = False


# In[42]:


if hvd.rank() == 0:
    print(model.summary())


# In[43]:


opt = tensorflow.keras.optimizers.Adam(0.0005 * hvd.size())
opt = hvd.DistributedOptimizer(opt)


# In[44]:


model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# In[45]:


if hvd.rank() == 0:
   verbose = 2
else:
   verbose=0


# In[46]:


x = model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(4, activation = "softmax")(x)


# In[26]:


modell_GPU = Model(inputs= baseModel.input , outputs = x)


# In[27]:


modell_GPU.summary()


# In[29]:


modell_GPU.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[30]:


modell_GPU.fit_generator(train_dataset,epochs=35)


# In[31]:


modell_GPU.save('modell_GPU.h5')


# In[ ]:




