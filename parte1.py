import cv2

cap = cv2.VideoCapture('img/moto1.mp4')
cap = cv2.VideoCapture('img/video1.mp4')

while cap.isOpened():
	_, frame = cap.read()
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()