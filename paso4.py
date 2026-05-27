#Cuando abres un archivo gráfico con CV2
#y quieres llevarlo a MATPLOTLIB
#debes transformar de BGR a RGB que maneja MATPLOTLIB
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('posgrado.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
#plt.imshow(img)
plt.show()