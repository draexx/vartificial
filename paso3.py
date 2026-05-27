from PIL import Image
from numpy import asarray

img = Image.open('boliviaok.png')

m = asarray(img)

print("Número de filas:", len(m))
print("Número de columnas:", len(m[0]))
c=0
for i in range(len(m)):
	for j in range(len(m[0])):
		print(m[i][j][0],m[i][j][1],m[i][j][2])
		c=c+1
print("Total Pixeles:",c)

