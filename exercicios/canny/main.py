import cv2
import numpy as np
import random
import requests
from io import BytesIO

# Constantes
STEP = 6
JITTER = 4
RAIO = 6
RAIO_PEQUENO = 3
IMAGE_URL = 'https://img.freepik.com/free-photo/futuristic-exploration-dubai-s-evolving-cityscape_23-2151339724.jpg?t=st=1719255845~exp=1719259445~hmac=5c6f5d7a28bc9bab8cf242a871d83a144ea92a9dc0f554221a554ffa78352225&w=1060'

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image_data = response.content
    return cv2.imdecode(np.array(bytearray(image_data), dtype=np.uint8), cv2.IMREAD_COLOR)

def apply_canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 80, 240)

def generate_pointillism(image, edge_image):
    height, width, _ = image.shape
    points = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    xrange = list(range(0, height, STEP))
    yrange = list(range(0, width, STEP))

    random.shuffle(xrange)
    random.shuffle(yrange)

    # Amostragem aleatória de pontos
    for i in xrange:
        for j in yrange:
            x = i + random.randint(-JITTER, JITTER)
            y = j + random.randint(-JITTER, JITTER)
            x = min(max(x, 0), height - 1)
            y = min(max(y, 0), width - 1)
            color = tuple(map(int, image[x, y]))
            cv2.circle(points, (y, x), RAIO, color, -1, cv2.LINE_AA)

    # Adiciona pontos baseados nas bordas
    border_points = []
    for i in range(height):
        for j in range(width):
            if edge_image[i, j] != 0:
                color = tuple(map(int, image[i, j]))
                border_points.append([j, i, color[0], color[1], color[2]])

    random.shuffle(border_points)

    for ponto in border_points:
        x, y, b, g, r = ponto
        color = (b, g, r)
        cv2.circle(points, (x, y), RAIO_PEQUENO, color, -1, cv2.LINE_AA)

    return points

def main():
    image = download_image(IMAGE_URL)
    
    cv2.imshow("Imagem Original", image)
    cv2.waitKey(0)

    edge_image = apply_canny_edge_detection(image)
    cv2.imshow("Bordas Canny", edge_image)
    cv2.waitKey(0)

    pointillism_image = generate_pointillism(image, edge_image)
    cv2.imshow("Imagem Pontilhista", pointillism_image)
    cv2.waitKey(0)

    cv2.imwrite("cannypoints.png", pointillism_image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
