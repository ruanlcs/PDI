import cv2

#algoritmo de classficação
#objetivo: eliminar objetos das bordas, eliminar as bolhas com buracos e no final a contagem de bolhas é 14

image = cv2.imread("bolhas.png", cv2.IMREAD_GRAYSCALE)

if image is None:
  print("Imagem não carregou corretamente")
  exit()

# Obtendo largura e altura da imagem
width = image.shape[1]
height = image.shape[0]
print(f"{width}x{height}")

# Excluir bordas
for i in range(height):
  if image[0,i] == 255:
    cv2.floodFill(image, None, (i,0), 0)
  if image[width - 1, i] == 255:
    cv2.floodFill(image, None, (i, width - 1), 0)

for i in range(width):
  if image[i,0] == 255:
    cv2.floodFill(image, None, (0,i), 0)
  if image[i, height - 1] == 255:
    cv2.floodFill(image, None, (height - 1, i),0)

# visulizar imagem sem bordas
cv2.imshow("Imagem_sembordas.png", image)
cv2.imwrite("Imagem_sembordas.png", image)

# Inicializando contagem de objetos
nobjects = 0

# Loop para encontrar objetos na imagem
for i in range(height):
  for j in range(width):
    if image[i,j] == 255:
      nobjects += 1
      seed_point = (j,i)
      cv2.floodFill(image, None, seed_point, nobjects)

# equalized = cv2.equalizeHist(image)
# cv2.imshow("imagem contada", image)
# cv2.imshow("realce", equalized)

# cv2.imwrite("image_realce.png", equalized)

#pintar fundo de branco para contagem de buracos
cv2.floodFill(image, None, (0,0), 255)

counte = 0
for i in range(height):
     for j in range(width):
      if image[i, j] == 0 and image[i, j - 1] > counte:
        counte += 1
        cv2.floodFill(image, None, (j - 1, i), counte)

cv2.imshow("Imagen-cont", image )     

counter = 0
for i in range(height):
  for j in range(width):
    if image[i, j] == 0: #and image[i, j - 1] > counter:
      #counter += 1
      cv2.floodFill(image, None, (j, i-1), 255)
      #cv2.floodFill(image, None, (j - 1, i), 255)





print(f"A figura tem {nobjects - counte} bolhas {counte}")

# Exibir a imagem com a segmentação
cv2.imshow("imagem", image)
cv2.imwrite("labeling.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
