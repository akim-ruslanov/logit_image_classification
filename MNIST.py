import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


dataset = MNIST(root='data/', download=True)
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())
print(dataset[0][0])