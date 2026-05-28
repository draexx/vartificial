import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import tensorflow as tf

# =====================================================================
# PASO 1: Cargar el modelo ya entrenado
# =====================================================================
# (Para este ejemplo, entrenamos un modelo rápido idéntico al anterior)
print("Cargando y entrenando el cerebro de la IA...")
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = np.expand_dims(train_images / 255.0, axis=-1)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3, batch_size=64, verbose=1)
print("¡IA Lista para usar!")

# =====================================================================
# PASO 2: Control de la Cámara con OpenCV
# =====================================================================
# Inicializa la cámara web (0 suele ser la cámara integrada de la laptop)
cap = cv2.VideoCapture(0)

print("\n--- INSTRUCCIONES ---")
print("1. Dibuja un número en negro sobre una hoja blanca (trazo grueso).")
print("2. Colócalo frente a la cámara.")
print("3. Presiona la BARRA ESPACIADORA para analizar.")
print("4. Presiona 'q' para salir.")

while True:
    # Capturar cuadro por cuadro de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Definir una "Zona de Interés" (un cuadrado en el centro de la pantalla)
    # Esto ayuda a que el usuario sepa dónde colocar el número escrito
    alto, ancho, _ = frame.shape
    x1, y1 = int(ancho/2 - 100), int(alto/2 - 100)
    x2, y2 = int(ancho/2 + 100), int(alto/2 + 100)
    
    # Dibujar el recuadro guía en la pantalla (Color verde, grosor 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Coloque el numero aqui", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar el video en vivo
    cv2.imshow('Detector de Numeros', frame)

    # Detectar teclas
    key = cv2.waitKey(1) & 0xFF
    
    # SI EL USUARIO PRESIONA LA BARRA ESPACIADORA: Procesar la imagen
    if key == ord(' '):
        # 1. Recortar solo lo que está dentro del recuadro verde
        roi = frame[y1:y2, x1:x2]
        
        # 2. Convertir a escala de grises
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Adaptar el contraste (Binarización)
        # MNIST tiene fondo negro y número blanco. Nuestra hoja es fondo blanco y número negro.
        # cv2.threshold con 'cv2.THRESH_BINARY_INV' invierte los colores automáticamente.
        _, binarizada = cv2.threshold(gris, 128, 255, cv2.THRESH_BINARY_INV)
        
        # 4. Redimensionar a 28x28 píxeles (el tamaño que exige la IA)
        miniatura = cv2.resize(binarizada, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 5. Normalizar los píxeles (0.0 a 1.0)
        img_preparada = miniatura / 255.0
        
        # 6. Adaptar las dimensiones para el modelo: (1, 28, 28, 1)
        img_preparada = np.expand_dims(img_preparada, axis=0)
        img_preparada = np.expand_dims(img_preparada, axis=-1)
        
        # 7. Ejecutar la predicción
        prediccion = model.predict(img_preparada)
        numero_detectado = np.argmax(prediccion[0])
        porcentaje_certeza = prediccion[0][numero_detectado] * 100
        
        # Mostrar el veredicto en la consola
        print(f"\n[ANALISIS DE IMAGEN]")
        print(f"-> ¡Estoy {porcentaje_certeza:.2f}% seguro de que es un: {numero_detectado}!")
        
        # Mostrar en una ventana pequeña cómo vio la IA tu número binarizado
        cv2.imshow('Como lo ve la IA (28x28)', cv2.resize(binarizada, (200, 200)))

    # SI EL USUARIO PRESIONA 'q': Salir del bucle
    elif key == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()