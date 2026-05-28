import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# PASO 1: Cargar y preparar los datos (Preprocesamiento)
# =====================================================================
print("--- Pasos 1: Cargando el dataset MNIST ---")
# El dataset ya viene dividido en datos de entrenamiento y de prueba
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalizar los píxeles para que estén en un rango de 0 a 1 (en lugar de 0 a 255)
# Esto ayuda a que la red neuronal converja y aprenda mucho más rápido.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Las CNNs esperan una estructura de 4 dimensiones: (num_imagenes, ancho, alto, canales_de_color)
# Como son imágenes en escala de grises, añadimos una dimensión extra al final (1 canal).
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

print(f"Imágenes de entrenamiento: {train_images.shape}")
print(f"Imágenes de prueba: {test_images.shape}\n")


# =====================================================================
# PASO 2 y 3: Definir la arquitectura de la Red Neuronal (CNN)
# =====================================================================
print("--- Pasos 2 y 3: Construyendo la estructura de la red ---")
model = models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    # Capa Convolucional 1: Extrae características básicas (bordes, líneas)
    # Usa 32 filtros de tamaño 3x3
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # Reduce el tamaño espacial de la imagen a la mitad
    
    # Capa Convolucional 2: Combina características para detectar formas más complejas
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Capa Convolucional 3: Refina los patrones encontrados
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Aplanado: Convertimos el mapa de características 2D en un vector 1D
    layers.Flatten(),
    
    # Capa oculta densa (Totalmente conectada) para interpretar los patrones
    layers.Dense(64, activation='relu'),
    
    # Capa de salida: 10 neuronas (una para cada dígito del 0 al 9)
    # Usamos activación 'softmax' para obtener probabilidades que sumen 100%
    layers.Dense(10, activation='softmax')
])

# Mostrar un resumen de la estructura de nuestra red
model.summary()


# =====================================================================
# PASO 4: Compilar y Entrenar el Modelo
# =====================================================================
print("\n--- Compilando el modelo ---")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Iniciando el entrenamiento (Esto puede tomar un minuto) ---")
# Entrenamos por 5 épocas (veces completas que la red revisará todo el dataset)
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))


# =====================================================================
# PASO 5: Evaluación y Simulación de una Predicción Real
# =====================================================================
print("\n--- Pasos 4 y 5: Evaluando el modelo con datos nuevos ---")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nPrecisión final en datos de prueba: {test_acc * 100:.2f}%')

# Vamos a tomar una imagen de prueba al azar para simular que el usuario la ingresa
indice_azar = 42  # Puedes cambiar este número para probar con otros dígitos
imagen_prueba = test_images[indice_azar]
etiqueta_real = test_labels[indice_azar]

# Para predecir, el modelo necesita un lote de imágenes, así que expandimos su dimensión
imagen_para_predecir = np.expand_dims(imagen_prueba, axis=0)

# El modelo ejecuta la predicción (distribución de probabilidad)
predicciones = model.predict(imagen_para_predecir)
digito_predicho = np.argmax(predicciones[0]) # Selecciona el índice con mayor probabilidad

print(f"\n[RESULTADO DE LA PREDICCIÓN]")
print(f"-> Matriz de probabilidades de la capa de salida:\n   {predicciones[0]}")
print(f"-> El programa predice que el número es un: {digito_predicho}")
print(f"-> El número real escrito es un: {etiqueta_real}")

# Opcional: Mostrar visualmente el resultado si tienes interfaz gráfica activa
plt.imshow(imagen_prueba.squeeze(), cmap='gray')
plt.title(f"Predicción: {digito_predicho} (Real: {etiqueta_real})")
plt.show()