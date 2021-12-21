import numpy
from PIL import Image

image = Image.open('pic3.jpg')
data = numpy.asarray(image)
print(data.shape)

#grayscale = numpy.array([numpy.array([pixel[0] for pixel in row]) for row in data])

def apply_impulsive_noise(image, amplitude, probability):
  for row in image:
    for pixel in row:
      if (numpy.random.choice([0, 1], 1, p=[1-probability, probability])):
        for byte in pixel:
          byte = 256
  return image

noisy = apply_impulsive_noise(data, 100, 0.2)

#restored = numpy.array([numpy.array(numpy.full((1,3), pixel) for pixel in row) for row in data])
restored_image = Image.fromarray(data)
restored_image.save("noisy.jpg", "JPEG")