import numpy as np
import pathlib as pl
from PIL import Image

def load_image(path: pl.Path) -> np.ndarray:
  """
  Load 'm x n x 3' RGB image to 'm x n' numpy ndarray
  """
  image = Image.open(path)
  data = np.asarray(image)
  out = np.ndarray((len(data), len(data[0])))
  for i in range(len(data)):
    for j in range(len(data[0])):
      out[i][j] = data[i][j][0]
  return out

def create_image(data: np.ndarray) -> Image:
  """
  Create 'm x n x 3' RGB image from 'm x n' numpy ndarray
  """
  arr = np.ndarray((len(data), len(data[0]), 3))
  for i in range(len(data)):
    for j in range(len(data[0])):
      arr[i][j][0] = data[i][j]
      arr[i][j][1] = data[i][j]
      arr[i][j][2] = data[i][j]
      if (data[i][j] > 255 or data[i][j] < 0):
        print(f"Found bad value: {i}, {j}, {data[i][j]}")
  image = Image.fromarray(arr.astype(np.uint8))
  return image

def apply_impulsive_noise(image_data: np.ndarray, amplitude, probability) -> np.ndarray:
  """
  Apply impulsive noise to image data
  """
  for i in range(len(image_data)):
    for j in range(len(image_data[0])):
      if (np.random.random() < probability):
          image_data[i][j] = (image_data[i][j] + amplitude) % 255
  return image_data

def get_mean(x: np.ndarray, i, j):
  """
  Calculate mean of pixel accounting for boundaries
  """
  mean = 0
  count = 1
  if i > 0 and j > 0:
    mean += x[i-1][j-1]
    count += 1
  if i > 0:
    mean += x[i-1][j]
    count += 1
  if i > 0 and j < len(x[0]) - 1:
    mean += x[i-1][j+1]
    count += 1
  if i < len(x) - 1 and j < len(x[0]) - 1:
    mean += x[i+1][j+1]
    count += 1
  if i < len(x) - 1:
    mean += x[i+1][j]
    count += 1
  if i < len(x) - 1 and j > 0:
    mean += x[i+1][j-1]
    count += 1
  if j < len(x[0]) - 1:
    mean += x[i][j+1]
    count += 1
  if j > 0:
    mean += x[i][j-1]
    count += 1
  mean += x[i][j]
  mean /= count
  return mean

def apply_filter(x: np.ndarray, beta, delta):
  """
  Apply filter
  """
  filtered = np.copy(x)
  for i in range(len(x)):
    for j in range(len(x[0])):

      m = get_mean(x, i, j)
      filtered[i][j] = m - ((beta + delta) / (2 * delta)) * abs( m + beta - x[i][j] ) \
                     + ((beta + delta) / (2 * delta)) * abs( m - beta - x[i][j]) \
                     - (beta) / (2 * delta) * abs(m - beta - delta - x[i][j]) \
                     + (beta) / (2 * delta) * abs(m + beta + delta - x[i][j])
      if filtered[i][j] > 255: filtered[i][j] = 255
      if filtered[i][j] < 0:   filtered[i][j] = 0

  return filtered

image = load_image("images/pic3.jpg")
noisy = apply_impulsive_noise(image, 100, 0.1)
filtered = apply_filter(noisy, 70, 1)
output = create_image(filtered)
output.save("images/filtered.jpg", "JPEG")
