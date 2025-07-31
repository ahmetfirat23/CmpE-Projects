## CMPE 597 Sp. Tp. Deep Learning
## Spring 2025 Assignment 1
### Tarik Can Ozden - Ahmet Firat Gamsiz - Group 10

This is the codebase for assignemnt 1. numpy, sklearn, matplotlib and pytorch are used in the project.

The custom activation functions, loss functions, forward and backward calls, SGD and data loader are implemented in ```utils.py```.

### Part 1. Classification with Cross-Entropy

For part 1, implementation from scratch code can be found in ```part1_model_custom.py```. The code can be run with ```python part1_model_custom.py```. Similarly, implementation with deep learning libraries part can be run with ```python part1_model_torch.py```, which is implemented with PyTorch.

### Part 2. Classification with Word Embeddings

Part 2 requires glove.6B embeddings to be downloaded from https://nlp.stanford.edu/projects/glove/. The glove 100-dimensioned embeddings should be placed inside ```glove.6B/glove.6B.100d.txt```.

For part 2, implementation from scratch code can be found in ```part2_model_custom.py```. The code can be run with ```python part2_model_custom.py```. Similarly, implementation with deep learning libraries part can be run with ```python part2_model_torch.py```, which is implemented with PyTorch.