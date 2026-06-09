# pyrefly: ignore [missing-import]
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

# Conexiones oficiales entre los 21 puntos de la mano
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Pulgar
    (0,5),(5,6),(6,7),(7,8),       # Índice
    (0,9),(9,10),(10,11),(11,12),  # Medio
    (0,13),(13,14),(14,15),(15,16),# Anular
    (0,17),(17,18),(18,19),(19,20),# Meñique
    (5,9),(9,13),(13,17)           # Palma
]


# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
MODELO = "./modelos/hand_landmarker.task"

COLOR_PUNTO  = (0, 255, 0)     # Verde
COLOR_LINEA  = (255, 255, 255)  # Blanco
RADIO_PUNTO  = 6
GROSOR_LINEA = 2

# ──────────────────────────────────────────────
# CREAR DETECTOR (Nueva API Tasks - modo video)
# ──────────────────────────────────────────────
opciones = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODELO),
    running_mode=RunningMode.VIDEO,   # Modo video para cámara en tiempo real
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(opciones)

# ──────────────────────────────────────────────
# CAPTURA DE CÁMARA EN TIEMPO REAL
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
timestamp_ms = 0  # La API de video requiere timestamps incrementales

print("--- DETECTOR DE MANOS EN VIVO (Tasks API) ---")
print("Muestra tu mano frente a la cámara.")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Efecto espejo (más natural)
    frame = cv2.flip(frame, 1)
    alto, ancho = frame.shape[:2]

    # Convertir frame de OpenCV (BGR) al formato MediaPipe (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Ejecutar detección pasando el timestamp actual en milisegundos
    resultado = detector.detect_for_video(imagen_mp, timestamp_ms)
    timestamp_ms += 33  # ~30 FPS

    # ──────────────────────────────────────────
    # DIBUJAR PUNTOS Y CONEXIONES
    # ──────────────────────────────────────────
    if resultado.hand_landmarks:
        cantidad = len(resultado.hand_landmarks)
        for landmarks_mano in resultado.hand_landmarks:
            # Convertir coordenadas normalizadas a píxeles
            puntos = [
                (int(lm.x * ancho), int(lm.y * alto))
                for lm in landmarks_mano
            ]

            # Dibujar conexiones (esqueleto)
            for inicio, fin in HAND_CONNECTIONS:
                cv2.line(frame, puntos[inicio], puntos[fin], COLOR_LINEA, GROSOR_LINEA)

            # Dibujar los 21 puntos clave
            for px, py in puntos:
                cv2.circle(frame, (px, py), RADIO_PUNTO, COLOR_PUNTO, -1)

        # Texto en pantalla con el conteo de manos
        cv2.putText(frame, f"Manos detectadas: {cantidad}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Sin manos detectadas", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Detector de Manos (Tasks API)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()