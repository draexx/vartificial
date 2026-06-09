# pyrefly: ignore [missing-import]
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

# Conexiones oficiales entre los 21 puntos de la mano (igual a HAND_CONNECTIONS)
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
MODELO  = "./modelos/hand_landmarker.task"
IMAGEN  = "./img/manos1.jpeg"
SALIDA  = "./resultados/res_manos1.jpeg"

# Constantes de color para dibujar manualmente
COLOR_PUNTO   = (0, 255, 0)    # Verde para los 21 puntos
COLOR_LINEA   = (255, 255, 255) # Blanco para las conexiones
RADIO_PUNTO   = 5
GROSOR_LINEA  = 2

# ──────────────────────────────────────────────
# CARGAR IMAGEN
# ──────────────────────────────────────────────
imagen_bgr = cv2.imread(IMAGEN)
if imagen_bgr is None:
    print(f"Error: No se pudo cargar '{IMAGEN}'")
    exit()

alto, ancho = imagen_bgr.shape[:2]

# MediaPipe Tasks necesita su propio formato de imagen
imagen_mp = mp.Image.create_from_file(IMAGEN)

# ──────────────────────────────────────────────
# CREAR DETECTOR (Nueva API Tasks)
# ──────────────────────────────────────────────
opciones = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODELO),
    running_mode=RunningMode.IMAGE,   # Modo imagen estática
    num_hands=2
)

with vision.HandLandmarker.create_from_options(opciones) as detector:
    # Ejecutar la detección
    resultado = detector.detect(imagen_mp)

    if not resultado.hand_landmarks:
        print("No se detectó ninguna mano en la imagen.")
        exit()

    print(f"¡Se detectaron {len(resultado.hand_landmarks)} mano(s)!")

    # ──────────────────────────────────────────
    # DIBUJAR PUNTOS Y CONEXIONES MANUALMENTE
    # ──────────────────────────────────────────
    for landmarks_mano in resultado.hand_landmarks:
        # Convertir coordenadas normalizadas (0.0–1.0) a píxeles
        puntos = [
            (int(lm.x * ancho), int(lm.y * alto))
            for lm in landmarks_mano
        ]

        # Dibujar las 21 conexiones (esqueleto de la mano)
        for inicio, fin in HAND_CONNECTIONS:
            cv2.line(imagen_bgr, puntos[inicio], puntos[fin], COLOR_LINEA, GROSOR_LINEA)

        # Dibujar los 21 puntos clave encima de las líneas
        for px, py in puntos:
            cv2.circle(imagen_bgr, (px, py), RADIO_PUNTO, COLOR_PUNTO, -1)

        # Imprimir coordenadas de la muñeca (punto 0)
        muneca = puntos[0]
        print(f"  Muñeca en píxeles: X={muneca[0]}, Y={muneca[1]}")

# ──────────────────────────────────────────────
# GUARDAR Y MOSTRAR RESULTADO
# ──────────────────────────────────────────────
cv2.imwrite(SALIDA, imagen_bgr)
cv2.imshow("Resultado - Detector de Manos (Tasks API)", imagen_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"\n[+] Imagen guardada en: {SALIDA}")