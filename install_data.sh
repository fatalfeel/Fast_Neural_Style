#!/bin/bash

sudo apt install unzip
sudo pip3 install --force-reinstall gdown

gdown http://images.cocodataset.org/zips/train2014.zip

mkdir -p ./dataset/train
unzip -j ./train2014.zip train2014/* -d ./dataset/train

gdown https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth --output ./pretrained/
gdown https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth --output ./pretrained/
