#!/usr/bin/env python
#coding: utf-8

# ----------------- extract-features.py -------------------
# Goal :
#   Extract features from image with keras/tensorflow (VGG16 model)
#
# Dependencies can be installed by running:
# 	pip install keras tensorflow h5py pillow
#
# Execution :
#   Written and tested using Python 2.7.13
#   Run script as:
#       ./extract-features.py ./input/images/*.jpg
#		printf '%s\0' ./input/images/*.jpg | xargs -0 ./extract-features.py
#		find ./input/images/ -maxdepth 1 -type f -name '*.jpg' -exec python ./extract-features.py '{}' +

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sys, json

def main():
    # Load model VGG16 / # build the VGG16 network
	# description in https://arxiv.org/abs/1409.1556
    # Takes ~6H with 2.4 GHz Intel Core 2 Duo / 8 GB 1067 MHz DDR3
    base_model = VGG16(weights='imagenet')
	print('Model loaded.')
    # Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    # (see the paper above for mode details)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    # For each image, extract the representation
    for image_path in sys.argv[1:]:
        features = extract_features(model, image_path)
        with open(image_path + ".json", "w") as out:
            json.dump(features, out)

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False)


def extract_features(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features.tolist()[0]

if __name__ == "__main__":
    main()
