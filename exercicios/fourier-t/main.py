import cv2
import numpy as np
import matplotlib.pyplot as plt

def swap_quadrants(image):
  image = image[:image.shape[0] & -2, :image.shape[1] & -2]

  centerX = image.shape[1] // 2
  centerY = image.shape[0] // 2

  A = image[:centerY, :centerX]
  B = image[:centerY, centerX:]
  C = image[centerY:, :centerX]
  D = image[centerY:, centerX:]

  tmp = A.copy()
  A[:] = D
  D[:] = tmp

  tmp = B.copy()
  B[:] = C
  C[:] = tmp

def calculate_magnitude_spectrum(image):
  dft_M = cv2.getOptimalDFTSize(image.shape[0])
  dft_N = cv2.getOptimalDFTSize(image.shape[1])
  padded = cv2.copyMakeBorder(image,0,dft_M - image.shape[0], 0, dft_N - image.shape[1], cv2.BORDER_CONSTANT, value=0)

  complex_image = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
  swap_quadrants(complex_image)

  magnitude = cv2.magnitude(complex_image[:,:,0], complex_image[:,:,1])
  magnitude +=1
  magnitude = np.log(magnitude)
  magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

  return magnitude

image = cv2.imread("senoide-256.png", cv2.IMREAD_GRAYSCALE)

magnitude_spectrum = calculate_magnitude_spectrum(image)

plt.subplot(121), plt.imshow(image, cmap = 'gray')
plt.title("image"), plt.axis("off")
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude spectrum"), plt.axis('off')
plt.show()