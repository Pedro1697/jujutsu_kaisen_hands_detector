# jujutsu_kaisen_hands_detector
Hands gesture detector in real time using deep learning and computer vision inspired by Jujutsu Kaisen anime

# Overview

This project implements a hands gesture detection based in RNN network with a LSTM layer to process the information, inspired by the domain expasion featured in the Jujutsu Kaisen anime.

# Main features

* Train a classification model using NN
* Real-time gesture detection through webcam
* Custom hand gesture dataset
* Modular scripts for easy implemantation

# Usage
1. Image capture
  * Open the photos.py script and modify the list of characters, num_sequences, sequence_length to build your own dataset
2. Build the dataset
  * Once all the data is collected you must use images_sets.py script to create the .pkl and .npz files for the model
3. Train the model: using the RNN_model.ipynb file 
  * Preprocess the data
  * Build the neural network
  * Train the model
  * Save the weights 
4. Real-time detection
  * With the weights loaded into the realtime_detection.py to start the detection

# Project Structure 
* Scripts/ # All the support scripts used in the project
* files/ # the dataset used divided in two and the weights of the model
* RNN_model.ipynb # Training notebook
* realtime_detection.py # Real-time detection script

# Technologies Used 
* Python 3.10+
* TensorFlow/keras
* OpenCV
* Numpy
* Jupyter Notebooks

# Credits
Inspired by Jujutsu Kaisen anime/manga
Nicholas Renotte for the project used as state of the art

Developed by Pedro1697


