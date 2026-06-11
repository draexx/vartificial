import cv2
# pyrefly: ignore [missing-import]
from cap_from_youtube import cap_from_youtube

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_url = "https://www.youtube.com/watch?v=I-TJKyVws58"
video_url = "https://www.youtube.com/watch?v=vwtMmpHcTGY"
cap = cap_from_youtube(video_url, 'best')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)

    caras = faceCascade.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(150, 150),      # más pequeño que cámara (rostros lejanos en video)
        maxSize=(500, 500)
    )

    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Rostro", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Rostros: {len(caras)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if len(caras) > 0 else (0, 0, 255), 2)

    cv2.imshow("Detector de Rostros - YouTube", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()