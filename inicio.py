import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from tensorflow.keras.datasets import mnist
import random

# Cargando el dataset MNIST
# 60000 para entrenamiento (train)
# 10000 para pruebas (test)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Selección de un índice aleatorio para mostrar imágenes dinámicamente en cada ejecución
idx = random.randint(0, len(X_train) - 1)
img = X_train[idx]
eti = y_train[idx]

plt.imshow(img, cmap='gray')
plt.title(f'Pedro Carranza - Índice: {idx} - Etiqueta: {eti}')
plt.axis('off')
plt.show()