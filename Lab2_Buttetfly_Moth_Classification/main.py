from torch.utils.tensorboard import SummaryWriter
from VGG19 import VGG19
from dataloader import BufferflyMothLoader
from ResNet50 import ResNet50
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
import os
from abc import ABC, abstractmethod
from BaseModel import BaseModel


if __name__ == "__main__":
    config = {
		"gpu": True,
        "device" : torch.device("cuda" if True and torch.cuda.is_available() else "cpu"),
		"batch_size": 32,
        "epoch_size": 100,
		"logdir": 'log/ResNet50_3/',
		"learning_rate": 0.0001, #2.5e-4, 1e-4, 1e-5
		"eval_interval": 1, #100
		"eval_episode": 3,
	}

    agent = BaseModel(100, config, 'ResNet50')

    # 訓練模型
    agent.train_model()
    # agent.load('./log/ResNet50_1/best_model_0.938.pth')
    # agent.print_train_test_accuracy()
    # 評估模型
    # agent.evaluate_model(is_valid=False)