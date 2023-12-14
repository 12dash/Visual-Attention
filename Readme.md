# Visual Attention Model

An implementation of Visual Attention model.  
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

Comments have been added to the .ipynb code.

## Dataset
Three datasets have been used for experimenting : 
* MNIST : Digit recognition classification task
* [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) | [Download](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) : Contains 10 classes for coloured imgaes
* [Cifar-100](https://www.cs.toronto.edu/~kriz/cifar.html)| [Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)  : Contains 100 classes under the label 'fina label' for coloured imgaes

The uncompressed data goes into the data folder

## Data Augmentation
Initially I tried to train the model without and data augmentation, but the model quickly overfit to the training. 
Hence I added the following data augmentation for cifar-10 and cifar-100:
```
transformation = v2.Compose([v2.RandomResizedCrop(size=(32, 32), antialias=True),
                             v2.RandomHorizontalFlip(p=0.5)])
```

## Experimentation
For all the models, the following architecture parameter was follows :
* Patch Size : 4 x 4
* Number of transformer layer : 12
* Number of multi-head attention : 8
* Embedding dim : 256
* Linear Projection dim : 512

### MNIST
For MNIST dataset, I trained it for only 5 epochs since the loss and accuracy had already reached quite a good value. 
<img src="result/mnist/predictions.png" width=30% height=30%>

### CIfar-10
<img src="result/cifar-10/predictions.png" width=30% height=30%>

### CIfar-100
<img src="result/cifar-100/predictions.png" width=30% height=30%>