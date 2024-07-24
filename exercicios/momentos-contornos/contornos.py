import cv2
import sys

def main(image_path):
    # Carrega a imagem em escala de cinza
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Não abriu {image_path}.")
        return

    # Aplica o limiar Otsu
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontra os contornos
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Converte a imagem para BGR
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Cria e escreve no arquivo SVG
    svg_path = "contornos.svg"
    with open(svg_path, 'w') as file:
        file.write(f'<svg height="{image.shape[0]}" width="{image.shape[1]}" xmlns="http://www.w3.org/2000/svg">\n')
        for contour in contours:
            if len(contour) > 0:
                file.write(f'<path d="M {contour[0][0][0]} {contour[0][0][1]} ')
                for point in contour[1:]:
                    file.write(f'L {point[0][0]} {point[0][1]} ')
                file.write('Z" fill="none" stroke="black" stroke-width="1"/>\n')
        file.write('</svg>\n')

    # Desenha os contornos na imagem
    cv2.drawContours(color_image, contours, -1, (0, 0, 255), 1)

    # Mostra a imagem
    cv2.imshow('Imagem', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Imprime o número de pontos gerados para cada contorno
    for i, contour in enumerate(contours):
        print(f"Contorno {i + 1}: {len(contour)} pontos")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python contornos.py <caminho/para/sua/imagem>")
    else:
        main(sys.argv[1])
