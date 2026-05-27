#Cuando abres un archivo gráfico con CV2
#y quieres llevarlo a MATPLOTLIB
#debes transformar de BGR a RGB que maneja MATPLOTLIB
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('naranja.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R, G, B = cv2.split(img_rgb)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(R, cmap='gray')
plt.title("Canal Rojo")
plt.subplot(1,3,2)
plt.imshow(G, cmap='gray')
plt.title("Canal Verde")
plt.subplot(1,3,3)
plt.imshow(B, cmap='gray')
plt.title("Canal Azul")
#plt.imshow(img_rgb)
#plt.imshow(img)
plt.show()