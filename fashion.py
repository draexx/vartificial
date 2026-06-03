import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images,trains_labels), (test_images,test_labels) = fashion_mnist_data.load_data()
train_images.shape
test_images.shape
train_images=train_images/255
test_images=test_images/255
img = train_images[0]
plt.imshow(img)
plt.show()
titulos=["Polera","Pantalon","Sueter","Vestido","Saco","Sandalia","Camisa","Zapatilla","Bolso","Botas"]
img = train_images[0]
plt.imshow(img)
plt.show()
print("Etiqueta: ",titulos[trains_labels[0]])

import random

# 1. Fijar la semilla de Python y NumPy
random.seed(42)
np.random.seed(42)

# 2. Fijar la semilla de TensorFlow
tf.random.set_seed(42)

# 3. (Opcional) Forzar operaciones deterministas en TensorFlow
tf.config.experimental.enable_op_determinism()
modelo = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10, activation="softmax")
])
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
historia = modelo.fit(train_images[...,np.newaxis],trains_labels,epochs=10,batch_size=256,verbose=2)
df = pd.DataFrame(historia.history)
df
#indice = np.random.choice(test_images.shape[0]) #1072
indice = 3322
print(indice)
imgPrueba = test_images[indice]
plt.imshow(imgPrueba)
plt.title(titulos[test_labels[indice]])
plt.show()
prediccion = modelo.predict(imgPrueba[np.newaxis, ...,np.newaxis])
#print(prediccion)
indicePred = np.argmax(prediccion[0])
print(indicePred, titulos[indicePred])
print(f"Prediccion del modelo: {titulos[indicePred]}")
print(f"Etiqueta real: {titulos[test_labels[indice]]}")