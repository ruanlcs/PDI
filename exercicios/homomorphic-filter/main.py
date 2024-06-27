import cv2
import numpy as np
import sys

def shift_dft(image):
  rows, cols = image.shape[:2]

  regions = [
    image[rows // 2: rows, cols // 2:cols],
    image[rows // 2:rows, 0:cols // 2],
    image[0:rows // 2, cols // 2:cols],
    image[0:rows // 2, 0:cols // 2]]
  
  a = np.hstack(regions[0:2])
  b = np.hstack(regions[2:4])

  return np.vstack((a,b))

def define_homomorphic_filter(rows, cols, d_0, c, gamma_high, gamma_low):
  homomorphic = np.zeros((rows, cols))

  rows_half = rows // 2
  cols_half = cols // 2

  for i in range(rows):
    i_diff = i - rows_half
    i_diff_square = i_diff * i_diff

    for j in range(cols):
      j_diff = j - cols_half
      j_diff_square = j_diff * j_diff

      d_2 = i_diff_square + j_diff_square
      exp_value = np.exp(-c*d_2/(d_0**2))
      homomorphic[i,j] = (gamma_high - gamma_low) * (1-exp_value) + gamma_low
  
  return cv2.merge([homomorphic, homomorphic])

def show_images(window_names, images, time):
  for image, window_name in zip(images, window_names):
    cv2.imshow(window_name, image)

  return chr(cv2.waitKey(time) & 255)

def options(key, c, gamma_high, gamma_low, d0):
  if key == "C":
    c += 0.5
  elif key == "c":
    c -= 0.5
    if c < 0:
      c =0
  elif key == "G":
    gamma_low += 0.5
  elif key == "g":
    gamma_low -= 0.5
    if gamma_low < gamma_low + 1:
      gamma_low = gamma_low + 1
  elif key == "D":
    d0 += 0.5
  elif key == "d":
    d0 -= 0.5

  return c, gamma_high, gamma_low, d0

def main():
  original_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
  original_image = cv2.resize(original_image, (1280, 720))

  dft_rows = cv2.getOptimalDFTSize(original_image.shape[0])
  dft_cols = cv2.getOptimalDFTSize(original_image.shape[1])

  padded_rows = dft_rows - original_image.shape[0]
  padde_cols = dft_cols - original_image.shape[1]

  original_window_name = "original image"
  filtered_window_name = "homomophirc filter"

  c = 1
  gamma_high = 2
  gamma_low = 0.5
  d0 = 4.5 

  homomorphic = define_homomorphic_filter(dft_rows, dft_cols, d0, c, gamma_high, gamma_low)

  padded = cv2.copyMakeBorder(original_image, 0, padded_rows, 0, padde_cols, cv2.BORDER_CONSTANT, value=[0,0,0]) 

  key = "não"

  while "\x1b" != key:
    planes = np.log(padded.astype(np.float64)+1)
    planes = cv2.merge([planes, np.zeros(padded.shape, np.float64)])

    dft_image = cv2.dft(planes, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    shifted_image = shift_dft(dft_image)

    filtered_image = cv2.mulSpectrums(shifted_image, homomorphic, 0)

    dft_image = shift_dft(filtered_image)

    image = cv2.idft(dft_image, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    key = show_images([original_window_name, filtered_window_name], [original_image, image], 0)

    c, gamma_high, gamma_low, d0 = options(key, c, gamma_high, gamma_low, d0)

    print("c {}, gamma high {}, gamma low {}, d0 {}".format(c, gamma_high, gamma_low, d0))

    homomorphic = define_homomorphic_filter(dft_rows, dft_cols, d0, c, gamma_high, gamma_low)

if __name__ == '__main__':
  main()
