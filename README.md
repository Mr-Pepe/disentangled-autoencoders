# Learning Interpretable Physical Variables from Image Sequences

This repository contains code to train Autoencoders for learning disentangled latent representations
of single images or image sequences in an unsupervised manner. It also contains code to generate a synthetic dataset
for this purpose. Evaluation functions give qualitative and quantitative feedback on the quality of 
disentanglement.

## Usage

Clone the repository and install the project and its dependencies.

    git clone https://github.com/Mr-Pepe/dl4cv
    cd dl4cv
    pip install -r requirements.txt
    pip install .
  
You can generate the dataset by running
    
    python generateData.py

in the dataset directory.

The models can be trained with, e.g.,

    python question_AE.py --train true
    
inside the final_runs directory.

The trained models can be evaluated with, e.g.,

    python question_AE.py --eval true


## Results

A sample of 2000 sequences was used to calculate the latent encoding for the three architectures.
Afterwards, walks over the latent variables were performed one by one.


### Question Autoencoder

![Alt text](doc/gifs/question_AE.gif) 

### Beta VAE

![Alt text](doc/gifs/beta_vae.gif) 


### Annealed VAE

![Alt text](doc/gifs/annealed_VAE.gif)
