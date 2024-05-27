# Fake/Real Face Classification 

## Overview
This project leverages a diverse set of four well-established classification models for classifying real and fake images: SqueezeNet, ShuffleNet, MobileNet-v3-large, and MobileNet-v3-small.

## Training 
Assuming you're in the code folder in the root directory, run:  

`python train.py --epochs=32 --batch_size=64 --save_interval=2 --dataset=easy lr=0.002`  

All parameters above are optional, so you can just run:  

`python train.py`  


## Testing 
## Evaluation  

`python train.py --epochs=32 --batch_size=64 --save_interval=2 --dataset=easy`  

All parameters in evaluation are optional too.