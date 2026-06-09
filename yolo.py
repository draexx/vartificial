# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import cv2
import imutils 
import os

modelo = YOLO('yolo11n.pt')

fuente="https://www.youtube.com/shorts/e6BX7utdm2s"

# O si deseas probar con una imagen local:
# fuente="./img/varios1.png"

resultados = modelo(fuente, stream=True)

# Crear la carpeta de resultados si no existe
carpeta_salida = "resultados"
os.makedirs(carpeta_salida, exist_ok=True)

# Determinar si la fuente es un video o stream
es_stream = "youtube.com" in fuente or "youtu.be" in fuente or any(ext in fuente.lower() for ext in [".mp4", ".avi", ".mov", ".mkv"])

frame_count = 0

for resultado in resultados:
    frame_count += 1
    
    if es_stream:
        # Si es stream, guardamos cada frame con su número secuencial
        ruta_salida = os.path.join(carpeta_salida, f"frame_{frame_count:04d}.jpg")
    else:
        # Si es una imagen estática, conservamos el nombre original
        nombre_archivo = os.path.basename(fuente)
        ruta_salida = os.path.join(carpeta_salida, f"res_{nombre_archivo}")
    
    # Guardar la imagen/frame anotada
    resultado.save(filename=ruta_salida)
    
    print(f"\n[+] [{frame_count}] Imagen anotada guardada en: {ruta_salida}")
    
    # Mostrar resumen de las detecciones en la terminal
    print("--- Detecciones ---")
    if len(resultado.boxes) == 0:
        print("  (Ningún objeto detectado)")
    else:
        for box in resultado.boxes:
            clase_id = int(box.cls[0])
            nombre_clase = modelo.names[clase_id]
            confianza = float(box.conf[0])
            print(f"- {nombre_clase.capitalize()}: {confianza:.2%}")
            
    # Si no es stream/video, terminamos después de procesar el único elemento
    if not es_stream:
        break
