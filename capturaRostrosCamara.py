import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)  # mejora contraste

    caras = faceCascade.detectMultiScale(
        gris,
        scaleFactor=1.05,     # más fino (antes 1.3)
        minNeighbors=3,       # más permisivo (antes 6)
        minSize=(150, 150),   # rostro grande porque estás cerca
        maxSize=(700, 700)    # permite rostros muy grandes
    )

    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Rostro", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Rostros: {len(caras)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0) if len(caras) > 0 else (0, 0, 255), 2)

    cv2.imshow("Detector de Rostros", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()