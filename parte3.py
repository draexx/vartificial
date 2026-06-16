import cv2
import numpy as np
from ultralytics import YOLO

coco_modelo = YOLO('yolo11n.pt')
placaDetecta = YOLO('l.pt')

vehiculos = {2: "auto", 3: "motocicleta", 5: "bus", 7: "camion"}

cap = cv2.VideoCapture('img/moto1.mp4')
#cap = cv2.VideoCapture('img/video1.mp4')
cap = cv2.VideoCapture('img/sample8.mp4')


ret, frame = cap.read()
rdi = cv2.selectROI("Selecciona la región de Interés",frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Selecciona la región de Interés")

x_rdi, y_rdi, w_rdi, h_rdi = rdi 

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	rdi_frame = frame[y_rdi:y_rdi+h_rdi,x_rdi:x_rdi+w_rdi].copy()
	resultados = coco_modelo.track(rdi_frame, persist=True)

	for resultado in resultados:
		if resultado.boxes.id is None:
			continue

		for box, clase, guia in zip(resultado.boxes.xyxy, resultado.boxes.cls, resultado.boxes.id):
			clase = int(clase)
			guia = int(guia)
			if clase in vehiculos:
				x1, y1, x2, y2 = map(int, box.cpu().numpy())
				x1 += x_rdi
				x2 += x_rdi
				y1 += y_rdi				
				y2 += y_rdi

				cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
				cv2.putText(frame, f'ID {guia} {vehiculos[clase]}', (x1,y1-10), 
					        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
				vehiculo_pedazo = frame[y1:y2, x1:x2].copy()
				res_placas = placaDetecta(vehiculo_pedazo)

				for resultado in res_placas:
					for box in resultado.boxes.xyxy:
						px1, py1, px2, py2 = map(int, box.cpu().numpy())
						px1 += x1 
						px2 += x1 
						py1 += y1
						py2 += y1
						cv2.rectangle(frame, (px1, py1), (px2,py2), (0,0,255),2)
						cv2.putText(frame, "Placa", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)

	cv2.imshow('Detección de Vehículos y Placas', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
