from matplotlib import pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray

plt.close('all')

model = tf.keras.models.load_model('model/model_300xepochs_fullscape',
                                   custom_objects=None,
                                   compile=True)

orig_img = imread('color_images/lena.png')
plt_img = resize(orig_img, (256,256))
plt.figure(), plt.imshow(plt_img), plt.title('Original image')
plt.figure(), plt.imshow(rgb2gray(plt_img), cmap='gray'), plt.title('Grayscale image')

img1_color=[]

img1=img_to_array(orig_img)
img1 = resize(img1, (256,256))
img1_color.append(img1)

img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))

output1 = model.predict(img1_color)
output1 = output1*128

result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]
img_color = lab2rgb(result)
plt.figure()
plt.imshow(img_color), plt.title('Colorized image')
