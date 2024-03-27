import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
import cv2

# Load the saved model
model = load_model('models/imageclassifierV2.h5')

img = cv2.imread('data/EggFertility/infertile/9i.jpeg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))
yhat

if yhat > 0.5: 
    print(f'Predicted class is InFertile')
else:
    print(f'Predicted class is Fertile')