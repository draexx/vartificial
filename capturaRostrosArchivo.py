import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("./img/rostro2.png")

caras = faceCascade.detectMultiScale(img, 
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30,30),
            maxSize=(200,200))

for (x, y,w,h) in caras:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
    
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, 1.1, 4)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#     cv2.imshow("frame",frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()