## CMPE 597 Sp. Tp. Deep Learning
## Spring 2025 Assignment 2
### Ahmet Firat Gamsiz 2020400180 - Tarik Can Ozden 2022400327 - Group 10

This is the codebase for assignment 2. Numpy, pickle, sklearn, matplotlib, pytorch and torchvision are used in the project.

### Part 1. Autoencoders

For part 1, implementation of autoencoders can be found in ```part1_autoencoders.py```. 
- Autoencoders are implemented as ```LSTMAutoencoder``` and ```ConvAutoencoder``` classes. 
- ```train_autoencoder``` function executes the training for the specified autoencoder and plots the loss curves.
- ```tsne_plot``` function plots the t-SNE figure for given model and image, label list.

The code can be executed with ```python part1_autoencoders.py```. 

### Part 2. Variational Autoencoders

For part 2, implementation for variational autoencoders can be found in ```part2_unconditional_vaes.py```. The code can be run with ```python part2_unconditional_vaes.py```.
- Sections of code are separated with comments for better readability.
- The code runs with grid search for hyperparameters. Refer to the section with ```Grid Search Setup``` for the parameters.
- Results are saved in ```outputs_unconditional``` folder. The best models are logged at the end of the training so please check your terminal for the best model. You can check the files by hand to find the folder with files starting with ```BEST_MODEL``` but that can be tedious.

Implementation of the conditional VAEs are in ```part2_conditional_vaes.py```. The code can be run with ```python part2_conditional_vaes.py```.
- Please make sure you have the classifier.pkl file in the same directory as the code. This file is used to generate the labels for the conditional VAE.
- Please first comment out the parameters of the model archiecture you want to use from ```HYPERPARAMETERS``` section in the code. Then, set the values of the hyperparameters you want to use in the ```HYPERPARAMETERS``` section. The code will run with the specified hyperparameters. Please note that there is no validation in this section and the final epoch is used for the final model.
- Results are saved in ```outputs_conditional``` folder. The confidence scores are logged at the end of the training so please check your terminal. This is the only way to check the confidence scores. The code does not save the confidence scores in a file.
