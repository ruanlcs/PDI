import cv2
import numpy as np
import requests

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    else:
        raise Exception("Erro ao baixar a imagem")

def kmeans_clustering(image, n_clusters=8, n_iterations=10, n_rodadas=1):
    samples = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 0.0001)
    flags = cv2.KMEANS_RANDOM_CENTERS

    for i in range(n_iterations):
        compactness, labels, centers = cv2.kmeans(samples, n_clusters, None, criteria, n_rodadas, flags)
        centers = np.uint8(centers)
        clustered_image = centers[labels.flatten()]
        clustered_image = clustered_image.reshape(image.shape)
        cv2.imshow("Clustered Image", clustered_image)
        cv2.imwrite(f"kmeans_image_{i}.png", clustered_image)
        print(f"Imagem {i + 1} salva como kmeans_image_{i}.png")

        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    image_url = "https://img.freepik.com/free-photo/seafood-plate-with-shrimps-mussels-lobsters-served-with-lemon_140725-8798.jpg?w=740&t=st=1719254482~exp=1719255082~hmac=36cdd489ec267993e8d9293c42b46cbe4b017f8b0548f6d258a71dbc001580dc"
    image = download_image(image_url)
    kmeans_clustering(image, n_clusters=8, n_iterations=10, n_rodadas=1)

if __name__ == "__main__":
    main()
