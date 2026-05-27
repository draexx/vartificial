import cv2

img = cv2.imread('posgrado.png')

cv2.imshow('Ventana principal',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
