# Brain Tumor MRI Image Classification
I have used MRI images to create a model to classify if the brain has tumor or not

## Installs
1) Tensorflow
2) Opencv-python
3) Matplotlib

## Data Exploration
1) Number of target categories
2) Example of images present in the dataset

## Image Preprocessing
1) Scaling
2) Gaussian Smoothning
3) Contrast
4) Brightness
5) Image Augmentation

## Model and Evaluation
1) <b>Custom Build Convolutional Neural Network:</b> A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

2) <b>Pre-trained InceptionResNetV2 model:</b> Inception-ResNet-v2 is a convolutional neural network that is trained on more than a million images from the ImageNet database. The network is 164 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images.
