import cv2
from ultralytics import YOLO

yolo_modelo = YOLO('yolo11x.pt')
yolo_modelo = YOLO('yolo11n.pt')

vehiculos = {2: "auto", 3: "motocicleta", 5: "bus", 7: "camion"}

cap = cv2.VideoCapture('img/moto1.mp4')
#cap = cv2.VideoCapture('img/video1.mp4')

while cap.isOpened():
	ret, frame = cap.read()

	resultados = yolo_modelo(frame)

	for resultado in resultados:
		for box, clase in zip(resultado.boxes.xyxy, resultado.boxes.cls):
			clase = int(clase)
			if clase in vehiculos:
				x1, y1, x2, y2 = map(int, box.cpu().numpy())
				cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
				cv2.putText(frame, vehiculos[clase], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
	cv2.imshow('Detección de Vehículos', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
