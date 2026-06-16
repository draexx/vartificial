from ultralytics import YOLO
import cv2

# =====================================================================
# CARGAR MODELO
# =====================================================================
modelo = YOLO('yolo11n.pt')

# =====================================================================
# CÁMARA EN TIEMPO REAL
# =====================================================================
cap = cv2.VideoCapture(0)

print("Detección YOLO en vivo. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Predecir en el frame actual
    resultados = modelo(frame, verbose=False)

    # Dibujar detecciones sobre el frame
    frame_anotado = resultados[0].plot()

    # Mostrar conteo de objetos detectados
    cantidad = len(resultados[0].boxes)
    cv2.putText(frame_anotado, f"Objetos: {cantidad}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if cantidad > 0 else (0, 0, 255), 2)

    # Imprimir detecciones en consola
    for box in resultados[0].boxes:
        clase_id  = int(box.cls[0])
        nombre    = modelo.names[clase_id]
        confianza = float(box.conf[0])
        print(f"  - {nombre.capitalize()}: {confianza:.2%}")

    cv2.imshow("YOLO - Detección en Vivo", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()