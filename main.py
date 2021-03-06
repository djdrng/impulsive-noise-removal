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

def get_median(x: np.ndarray, i, j):
  """
  Calculate median of pixel accounting for boundaries
  """
  med_vals = []
  if i > 0 and j > 0:
    med_vals.append(x[i-1][j-1])
  if i > 0:
    med_vals.append(x[i-1][j])
  if i > 0 and j < len(x[0]) - 1:
    med_vals.append(x[i-1][j+1])
  if i < len(x) - 1 and j < len(x[0]) - 1:
    med_vals.append(x[i+1][j+1])
  if i < len(x) - 1:
    med_vals.append(x[i+1][j])
  if i < len(x) - 1 and j > 0:
    med_vals.append(x[i+1][j-1])
  if j < len(x[0]) - 1:
    med_vals.append(x[i][j+1])
  if j > 0:
    med_vals.append(x[i][j-1])
  med_vals.append(x[i][j])
  med_vals.sort()
  return med_vals[len(med_vals) // 2]

def apply_median_filter(x: np.ndarray):
  """
  Apply median filter, for comparison
  """
  filtered = np.copy(x)
  for i in range(len(x)):
    for j in range(len(x[0])):
      filtered[i][j] = get_median(x, i, j)
  return filtered

def apply_rudimentary_filter(x: np.ndarray, beta):
  """
  Apply rudimentary version of the filter, with only a beta value
  """
  filtered = np.copy(x)
  for i in range(len(x)):
    for j in range(len(x[0])):

      m = get_median(x, i, j)
      if filtered[i][j] > m + beta:
        filtered[i][j] = m + beta
      elif filtered[i][j] < m - beta:
        filtered[i][j] = m - beta

      if filtered[i][j] > 255: filtered[i][j] = 255
      if filtered[i][j] < 0:   filtered[i][j] = 0

  return filtered

def apply_filter(x: np.ndarray, beta, delta):
  """
  Apply filter
  """
  filtered = np.copy(x)
  for i in range(len(x)):
    for j in range(len(x[0])):

      m = get_median(x, i, j)
      filtered[i][j] = m - ((beta + delta) / (2 * delta)) * abs( m + beta - x[i][j] ) \
                     + ((beta + delta) / (2 * delta)) * abs( m - beta - x[i][j]) \
                     - (beta) / (2 * delta) * abs(m - beta - delta - x[i][j]) \
                     + (beta) / (2 * delta) * abs(m + beta + delta - x[i][j])
      if filtered[i][j] > 255: filtered[i][j] = 255
      if filtered[i][j] < 0:   filtered[i][j] = 0

  return filtered

def calculate_image_difference(x: np.ndarray, y: np.ndarray):
  """
  Find's the average RGB difference per pixel for the two images.
  In other words, calculates a "score" for a filter to see how
  the filtered image compares to the original
  """
  assert len(x) == len(y)
  assert len(x[0]) == len(y[0])

  diff = 0
  for i in range(len(x)):
    for j in range(len(x[0])):
      diff += abs(x[i][j] - y[i][j]) 

  return diff / (len(x) * len(x[0]))


image_list = ["pic1", "pic2", "pic3"]
beta_values = [45, 50, 55, 60, 65, 70, 75]
#beta_values = [20, 40, 60, 80]
delta_values = [1]

for image_name in image_list:
  file_name = "images/" + image_name + ".jpg"
  original = load_image(file_name)
  image = np.copy(original)

  # Add noise to the image
  noisy = apply_impulsive_noise(image, 100, 0.1)
  noisy_image = create_image(noisy)
  noisy_image.save("images/" + image_name + "_noisy.jpg", "JPEG")

  # Apply Median Filter for contrast
  noisy_data = np.copy(noisy)
  median = apply_median_filter(noisy_data)
  median_image = create_image(median)
  median_image.save("images/" + image_name + "_median.jpg", "JPEG")
  median_results = calculate_image_difference(original, median)
  print(f"{image_name : <10} Median Results: {median_results}")

  best_result = 255
  best_beta = 0
  best_delta = 0
  best_rudimentary_result = 255
  best_rudimentary_beta = 0

  for beta in beta_values:
    # Test rudimentary filter
    noisy_data = np.copy(noisy)
    rudimentary = apply_rudimentary_filter(noisy_data, beta)
    rudimentary_image = create_image(rudimentary)
    rudimentary_results = calculate_image_difference(original, rudimentary)
    if rudimentary_results < best_rudimentary_result:
      best_rudimentary_result = rudimentary_results
      best_rudimentary_beta = beta

    print(f"{image_name : <10} {beta : ^10} Rudimentary Results: {rudimentary_results}")
  
    for delta in delta_values:
      # Normal Filter
      noisy_data = np.copy(noisy)
      filtered = apply_filter(noisy_data, beta, delta)
      filtered_image = create_image(filtered)
      results = calculate_image_difference(original, filtered)
      if results < best_result:
        best_result = results
        best_beta = beta
        best_delta = delta

      print(f"{image_name : <10} {beta : ^10} {delta : ^10} Results: {results}")

  # Select the best results and save the images
  noisy_data = np.copy(noisy)
  rudimentary = apply_rudimentary_filter(noisy_data, best_rudimentary_beta)
  rudimentary_image = create_image(rudimentary)
  rudimentary_image.save("images/" + image_name + "_rudimentary.jpg", "JPEG")

  noisy_data = np.copy(noisy)
  filtered = apply_filter(noisy_data, best_beta, best_delta)
  filtered_image = create_image(filtered)
  filtered_image.save("images/" + image_name + "_filtered.jpg", "JPEG")


# 5% noise probability to show increase in performance
file_name = "images/pic3.jpg"
original = load_image(file_name)

# Add noise to the image
noisy = apply_impulsive_noise(original, 100, 0.05)
filtered = apply_filter(noisy, 60, 1)
filtered_image = create_image(filtered)
filtered_image.save("images/pic3_5_filtered.jpg", "JPEG")
