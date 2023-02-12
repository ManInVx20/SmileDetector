# Smile Detector

A real-time smile detector program.

## Description

Main program captures real-time video from webcam. Then, detects faces and draw a rectangle around the face area. When the person smiles, it should print a text "Smiling" above the rectangle area. When not smiling, it should print a text "Not Smiling".

## How it works

* Get smiling and not smiling dataset images
* Train a network in the dataset
* Evaluate the network
* Detect face with Haar Cascade
* Extract the Region of Interest (ROI)
* Pass the ROI through trained network
* Output the result from trained network

## Project structure

    .
    ├── cascade                 # Cascade classifiers
    ├── datasets                # Data for fitting model
    ├── model                   # Output from training model
    ├── src                     # Codes for training and detection
    ├── videos                  # Output from detection
    ├── README.md
    └── requirements.txt

## Usage

Pre-requisite: python and pip

Install necessary packages from `requirements.txt` file
```bash
pip install -r requirements.txt
```

Run the model training
```bash
python src/train_model.py --dataset datasets/SMILEsmileD --model model/lenet.hdf5
```

Run the smile detection using CNN
```bash
python src/detect_smile.py --cascade cascade/haarcascade_frontalface_default.xml --model model/lenet.hdf5
```

Run the smile detection using Haar cascade
```bash
python src/detect_smile_cascade.py
```

# Reference

* https://www.pyimagesearch.com/2021/07/14/smile-detection-with-opencv-keras-and-tensorflow/
* https://d2l.aivivn.com/chapter_convolutional-neural-networks/lenet_vn.html