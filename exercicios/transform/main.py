import cv2
import numpy as np

def apply_morphology(image, str_element):
    # Aplica dilatação seguida de erosão
    dilated = cv2.dilate(image, str_element)
    eroded = cv2.erode(dilated, str_element)
    # Inverte a imagem para obter o resultado final desejado
    return cv2.bitwise_not(eroded)

def main():
    # Carrega as imagens
    images = []
    for i in range(1, 6):
        filename = f"digitos-{i}.png"
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Erro ao carregar a imagem: {filename}")
            return -1
        images.append(img)

    # Inverte as cores das imagens
    for i in range(5):
        images[i] = cv2.bitwise_not(images[i])

    # Elemento estruturante
    str_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

    # Aplica operações de morfologia para cada imagem
    result_images = []
    for i in range(5):
        result = apply_morphology(images[i], str_element)
        result_images.append(result)
        cv2.imwrite(f"morfologia{i+1}.png", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
