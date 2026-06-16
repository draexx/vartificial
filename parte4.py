import cv2
import numpy as np 
import csv
import os 
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
from paddleocr import PaddleOCR
# pyrefly: ignore [missing-import]
from cvzone.Utils import cornerRect, putTextRect

ocr = PaddleOCR(use_textline_orientation=True, lang="en")

coco_modelo = YOLO("yolo11x.pt")
placaDetecta = YOLO("l.pt")

csv_archivo = "placas.csv"

if not os.path.exists(csv_archivo):
    with open(csv_archivo, "w", newline='') as archivo:
        grabar = csv.writer(archivo)
        grabar.writerow(['nroFrame', 'id_v', 'placa'])

cap = cv2.VideoCapture('img/sample4.mp4')

vehiculos = {2: "auto", 3: "motocicleta", 5: "bus", 7: "camion"}

nroFrame = 0
bt_guia = "bytetrack.yaml"

# ── Bug 1: placasV debe ser dict, no sobreescribirse con resultado YOLO ──
placasV = {}

ret, frame = cap.read()
if not ret:
    print("Error al leer el archivo!!!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

rdi = cv2.selectROI("Selecciona la región de Interés", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Selecciona la región de Interés")
x_rdi, y_rdi, w_rdi, h_rdi = rdi
if w_rdi == 0 or h_rdi == 0:
    print("Selección de ROI cancelada o inválida. Saliendo...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    nroFrame += 1
    rdi_frame = frame[y_rdi:y_rdi+h_rdi, x_rdi:x_rdi+w_rdi].copy()

    resultados = coco_modelo.track(rdi_frame, persist=True, tracker=bt_guia, classes=list(vehiculos.keys()))

    guiaVehiculos = {}

    if resultados[0].boxes.id is not None:
        for box, guia, clase in zip(resultados[0].boxes.xyxy,
                                     resultados[0].boxes.id,
                                     resultados[0].boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy()
            clase = int(clase)
            guia  = int(guia)  # Bug 2: guia es el ID del vehículo para tracking

            x1 += x_rdi; x2 += x_rdi
            y1 += y_rdi; y2 += y_rdi

            # Bug 3: guardar por ID (guia), no por clase
            guiaVehiculos[guia] = (x1, y1, x2, y2)

    # Bug 4: variable "placasV" sobreescrita — renombrar resultado YOLO
    resultado_placas = placaDetecta(rdi_frame)[0]

    for resultado in resultado_placas.boxes.data.tolist():
        x1, y1, x2, y2, valor, qTipo = resultado

        x1 += x_rdi; x2 += x_rdi
        y1 += y_rdi; y2 += y_rdi

        for id_vehiculo, (xauto1, yauto1, xauto2, yauto2) in guiaVehiculos.items():
            if x1 > xauto1 and y1 > yauto1 and x2 < xauto2 and y2 < yauto2:

				# ── RECUADRO ROJO DE LA PLACA ──
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, "Placa", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                pedazo = frame[int(y1):int(y2), int(x1):int(x2)]
                if pedazo.size == 0:   # Bug 5: evitar crash con recorte vacío
                    continue

                pedazo = cv2.resize(pedazo, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
                pedazo_gris = cv2.cvtColor(pedazo, cv2.COLOR_BGR2GRAY)
                pedazo_gris = cv2.cvtColor(pedazo_gris, cv2.COLOR_GRAY2BGR)

                ocr_resultado = ocr.predict(pedazo_gris)

                placa_texto = ""

                if ocr_resultado and len(ocr_resultado) > 0:
                    res = ocr_resultado[0]
                    for texto, conf in zip(res.get('rec_texts', []), res.get('rec_scores', [])):
                        if conf > 0.7:
                            placa_texto = texto.upper().replace(" ", "")
                            break

                if id_vehiculo not in placasV or len(placa_texto) > len(placasV.get(id_vehiculo, "")):
                    placasV[id_vehiculo] = placa_texto

                texto_placa = placasV.get(id_vehiculo, "")

                with open(csv_archivo, 'a', newline='') as archivo:
                    grabar = csv.writer(archivo)
                    # Bug 7: usaba variable "base" que no existía
                    grabar.writerow([nroFrame, id_vehiculo, texto_placa])

                cornerRect(frame, (int(xauto1), int(yauto1),
                                   int(xauto2-xauto1), int(yauto2-yauto1)),
                           l=10, rt=2, colorR=(255, 0, 0))

                putTextRect(frame, f'Vehiculo {id_vehiculo}',
                            (int(xauto1), int(yauto1)-10),
                            scale=0.8, thickness=2,
                            colorR=(255,0,0), colorB=(255,255,255))

                if texto_placa:
                    putTextRect(frame, texto_placa,
                                (int(x1), int(y1)-10),
                                scale=1.3, thickness=2,
                                colorR=(0,0,0), colorB=(255,255,255), border=3)

    cv2.imshow('Detección de Vehículos y Placas', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()