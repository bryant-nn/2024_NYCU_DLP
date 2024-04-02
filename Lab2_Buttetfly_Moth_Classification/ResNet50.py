import torch
import torch.nn as nn
# from BaseModel import BaseModel

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        print("ResNet50 model initialized")
    def forward(self, x):
        print