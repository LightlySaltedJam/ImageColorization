# Image Colorization using Deep Neural Networks
Machine Learning - Black and White Image Colorization - Python  
Code inspired from  
https://github.com/bnsreenu/python_for_microscopists/blob/master/090a-autoencoder_colorize_V0.2.py  
https://github.com/erykml/video_games_colorization/blob/master/cnn_keras_test.ipynb  

## Examples

![alt text](https://github.com/LightlySaltedJam/ImageColorization/blob/main/Examples/Example1.PNG)
![alt text](https://github.com/LightlySaltedJam/ImageColorization/blob/main/Examples/Example2.PNG)
![alt text](https://github.com/LightlySaltedJam/ImageColorization/blob/main/Examples/Example4.PNG)
![alt text](https://github.com/LightlySaltedJam/ImageColorization/blob/main/Examples/Example3.PNG)

The model was trained on database of mostly landscape images. Therefore it handles images of hills, mountains, skies and forests quite well, but struggles when we test it on pictures of people, for example.  
Proposed solution: use a bigger, more diverse dataset. Problem: training time increases

## Project Setup 
### In order to run, the folder structure should look something like this:
<ul>
  <li>color_images</li>
  <ul>
    <li>8.jpg</li>
    <li>9.jpg</li>
    <li>12.jpg</li>
    <li>312.jpg</li>
    <li>lena.png</li>
  </ul>
  <li>models</li>
  <ul>
    <li>model_150xepochs++.h5</li>
    <li>model_300xepochs_fullscape.h5</li>
  </ul>
<li>Autoencoder_Alg.py</li>
<li>Autoencoder_Colorize.py</li>


## File Info

<li>Autoencoder_Alg.py will run the machine learning algorithm and start training. **Do not run without setting up a dataset** </li>
<li>Autoencoder_Colorize.py twill colorize a given image from the color_images folder using a set model. By default it will run model_300xepochs_fullscape.h5</li>


