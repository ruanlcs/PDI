import cv2
import numpy as np

def main(image_path):
    # Carrega a imagem em escala de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Não abriu {image_path}.")
        return

    # Aplica limiarização
    _, thresholded = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # Encontra contornos
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Converte a imagem de volta para BGR para desenhar colorido
    image_color = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

    nformas = 0
    total_pontos = 0

    for i, contour in enumerate(contours):
        if len(contour) < 10:
            continue

        nformas += 1
        total_pontos += len(contour)  # Conta o número de pontos no contorno

        momentos = cv2.moments(contour)
        center = (int(momentos['m10'] / momentos['m00']), int(momentos['m01'] / momentos['m00']))

        hu = cv2.HuMoments(momentos).flatten()
        if hu[0] > 0:
            cv2.drawContours(image_color, [contour], -1, (0, 0, 255), 2)
        else:
            cv2.drawContours(image_color, [contour], -1, (0, 255, 0), 2)

        cv2.putText(image_color, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 8)
        cv2.putText(image_color, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"Número de objetos: {nformas}")
    print(f"Total de pontos nos contornos: {total_pontos}")

    # Mostra a imagem resultante
    cv2.imshow("janela", image_color)
    cv2.imwrite("contornos-rotulados.png", image_color)
    cv2.waitKey()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python script.py <caminho/para/sua/imagem>")
    else:
        main(sys.argv[1])