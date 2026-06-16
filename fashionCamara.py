import cv2
import tensorflow as tf
import numpy as np

# =====================================================================
# PASO 1: Entrenar el modelo
# =====================================================================
titulos = ["Polera","Pantalon","Sueter","Vestido","Saco",
           "Sandalia","Camisa","Zapatilla","Bolso","Botas"]

print("Cargando y entrenando modelo...")
(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0

modelo = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
modelo.fit(train_images[..., np.newaxis], train_labels,
           epochs=10, batch_size=256, verbose=2)
print("¡Modelo listo!")

# =====================================================================
# PASO 2: Cámara en tiempo real
# =====================================================================
cap = cv2.VideoCapture(0)

print("\n--- INSTRUCCIONES ---")
print("Muestra una prenda de vestir frente a la cámara.")
print("Presiona ESPACIO para analizar.")
print("Presiona 'q' para salir.")

resultado_texto = ""
confianza_texto = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape

    # Zona de interés centrada
    x1, y1 = ancho//2 - 100, alto//2 - 100
    x2, y2 = ancho//2 + 100, alto//2 + 100
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Coloque la prenda aqui", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar último resultado en pantalla
    if resultado_texto:
        cv2.putText(frame, resultado_texto, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, confianza_texto, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    cv2.imshow("Detector de Prendas", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESPACIO: analizar lo que hay en el recuadro
    if key == ord(' '):
        roi = frame[y1:y2, x1:x2]

        # Preprocesar igual que Fashion MNIST
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gris = cv2.bitwise_not(gris)          # invertir: fondo negro, prenda blanca
        miniatura = cv2.resize(gris, (28, 28))
        normalizada = miniatura / 255.0
        entrada = normalizada[np.newaxis, ..., np.newaxis]

        # Predecir
        prediccion = modelo.predict(entrada, verbose=0)
        indice = np.argmax(prediccion[0])
        confianza = prediccion[0][indice] * 100

        resultado_texto = f"Prenda: {titulos[indice]}"
        confianza_texto = f"Confianza: {confianza:.1f}%"

        print(f"\n[ANALISIS] {titulos[indice]} — {confianza:.1f}% seguro")

        # Mostrar recorte procesado
        cv2.imshow("Como lo ve la IA (28x28)", cv2.resize(miniatura, (200, 200)))

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()