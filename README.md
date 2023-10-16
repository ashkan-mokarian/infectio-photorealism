# Infectio-photorealistic
Photorealistic output for [Infectio](http://). Code to create training dataset
and training models to transfer the simplistic x,y positions output of simulation
runs in Infectio to realistic looking videos of virus spread in plaque experiements.

# Demo
Shows result for model trained for around one milion steps. Left: GT, Middle:
model output, right: GT - ouput
![Result comparison](./attachments/M061_14.gif)

# How to use

## Dataset

## Training
Set train parameters in config.py, then run `python train.py`.

### on Slurm (hemera)

## Prediction

# To Do

# References
reference for pix2pix paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)