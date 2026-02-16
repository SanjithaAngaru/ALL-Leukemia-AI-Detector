'''This code was originally done on a simple platform called Google Colab, to make it more accessible, I transfered the following code here.'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Install and use the official 2026 Legacy Bridge
!pip install -q tf-keras
import tf_keras as keras
from tf_keras.layers import DepthwiseConv2D


# 2. THE SAFETY PATCH (Prevents RecursionError)
if not hasattr(DepthwiseConv2D, '_patched'):
   original_from_config = DepthwiseConv2D.from_config
   @classmethod
   def patched_from_config(cls, config):
       config.pop('groups', None) # Remove 2026-incompatible argument
       return original_from_config(config)
   DepthwiseConv2D.from_config = patched_from_config
   DepthwiseConv2D._patched = True
   print("Patch applied safely.")


# 3. Load Model (Ensure you re-uploaded 'keras_model.h5' to the sidebar)
if os.path.exists('/content/keras_model.h5'):
   model = keras.models.load_model('/content/keras_model.h5', compile=False)
   print("Model loaded successfully!")
else:
   print("ERROR: Please re-upload 'keras_model.h5' to the sidebar folder.")


# 4. Saliency Map Function (Most stable way for Teachable Machine)
def get_saliency_map_final(img_path, model):
   img = cv2.imread(img_path)
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img_resized = cv2.resize(img_rgb, (224, 224))
   x = np.expand_dims(img_resized, axis=0).astype(np.float32)
   x = (x / 127.5) - 1


   images = tf.Variable(x)
   with tf.GradientTape() as tape:
       tape.watch(images)
       preds = model(images)
       class_idx = np.argmax(preds)
       loss = preds[0, class_idx]


