import cv2
import numpy as np
import matplotlib.pyplot as plt

def swap_quadrants(image):
    tmp = np.copy(image)
    cx = image.shape[1] // 2
    cy = image.shape[0] // 2
    image[:cy, :cx] = tmp[cy:, cx:]
    image[cy:, cx:] = tmp[:cy, :cx]
    image[:cy, cx:] = tmp[cy:, :cx]
    image[cy:, :cx] = tmp[:cy, cx:]

def main():
    # Carrega a imagem em ponto flutuante a partir do arquivo YAML
    fs = cv2.FileStorage("senoide-256.yml", cv2.FILE_STORAGE_READ)
    image = fs.getNode("mat").mat()
    fs.release()

    # Calcula a transformada de Fourier
    complex_image = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    swap_quadrants(complex_image)

    # Separa os planos real e imaginário
    planes = cv2.split(complex_image)

    # Calcula o espectro de magnitude
    magnitude = cv2.magnitude(planes[0], planes[1])
    magnitude += 1
    cv2.log(magnitude, magnitude)

    # Normaliza a magnitude
    cv2.normalize(magnitude, magnitude, 0, 1, cv2.NORM_MINMAX)

    # Exibe as imagens processadas
    cv2.imshow("Imagem", image.astype(np.uint8))
    cv2.imshow("Espectro de magnitude", magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()