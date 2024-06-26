from torch.utils.tensorboard import SummaryWriter
from dataloader import BufferflyMothLoader
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import os
from abc import ABC, abstractmethod
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from VGG19 import VGG19
from ResNet50 import ResNet50

class BaseModel():
    def __init__(self, num_classes, config, model_name):
        super(BaseModel, self).__init__()

        self.gpu = config["gpu"]
        self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        self.total_time_step = 0
        self.config = config

        self.batch_size = int(config["batch_size"])
        self.epoch_size = int(config["epoch_size"])

        self.eval_interval = config["eval_interval"]
        self.eval_episode = config["eval_episode"]

        self.learning_rate = config["learning_rate"]
        self.model_name = model_name
        if model_name == 'VGG19':
            self.model = VGG19(num_classes, config).to(self.device)
        else:
            self.model = ResNet50(num_classes).to(self.device)

        self.max_valid_accuracy = np.NINF
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.1, verbose=True)
        self.criterion = nn.CrossEntropyLoss()
   
    # def configure_optimizers(self):
    #     # for name, param in self.named_parameters():
    #     #     print(name)
    #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    #     # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, verbose=True)
    #     self.criterion = nn.CrossEntropyLoss()
    #     return self.optimizer
    
    # @abstractmethod
    # def forward(self, x):
    #     return NotImplementedError
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def evaluate_model(self, dataset, is_valid=False):
        self.model.eval()
        if is_valid:
            validate_data = BufferflyMothLoader('./dataset/', dataset)
            validate_loader = data.DataLoader(dataset=validate_data, batch_size=self.batch_size, shuffle=False)
        else:
            validate_data = BufferflyMothLoader('./dataset/', dataset)
            validate_loader = data.DataLoader(dataset=validate_data, batch_size=self.batch_size, shuffle=False)

        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        eval_accuracy = correct_predictions / total_predictions
        loss = loss.sum().item() / total_predictions

        if is_valid and eval_accuracy > self.max_valid_accuracy and eval_accuracy > 0.85:
            self.max_valid_accuracy = eval_accuracy
            torch.save(self.model.state_dict(), os.path.join(self.logdir, f'best_model_{eval_accuracy}.pth'))
        
        if not is_valid:
            print(f'Test Accuracy: {eval_accuracy}')
        
        return eval_accuracy, loss
    
    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        

    def test_model(self):
        self.model.eval()
        test_data = BufferflyMothLoader('./dataset/', 'test')
        test_loader = data.DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)

        correct_predictions = 0
        total_predictions = 0


    def train_model(self):
        # 創建一個 TensorBoard 寫入器
        self.writer = SummaryWriter(self.config["logdir"])

        self.logdir = self.config["logdir"]
        # self.configure_optimizers()
        self.model.train()  # 設置模型為訓練模式

        train_data = BufferflyMothLoader('./dataset/', 'train')
        train_loader = data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch_size):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()  # 將數據移到 GPU（如果可用）
                self.optimizer.zero_grad()  # 重置梯度
                outputs = self.model(inputs)  # 前向傳播
                # print('outputs:', outputs, 'labels:', labels)
                # print(outputs.shape, labels.shape)
                loss = self.criterion(outputs, labels)  # 計算損失
                # print('loss', loss)


                # 計算訓練準確率
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                loss.backward()  # 反向傳播
                # for param in self.parameters():
                #     print(param.grad.norm())
                
                self.optimizer.step()  # 更新參數
                # for name, param in self.model.named_parameters():
                #     print(name, param.grad)

                # for group in self.optimizer.param_groups:
                #     for param in group['params']:
                #         print(param.data)


                running_loss += loss.item() * inputs.size(0)

            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_train_accuracy = correct_predictions / total_predictions

            
            # 寫入訓練損失到 TensorBoard
            self.writer.add_scalar('Train/Epoch Loss', epoch_loss, epoch)
            # self.writer.add_scalar('Train/Epoch Accuracy', epoch_train_accuracy, epoch)

            print(f'Epoch {epoch + 1} / {self.epoch_size}, Loss: {epoch_loss}, Accuracy: {epoch_train_accuracy}')

            # validate model
            eval_loss = 0
            if epoch % self.eval_interval == 0:
                eval_accuracy, eval_loss = self.evaluate_model(is_valid=True, dataset='test')
                self.writer.add_scalars('Accuracy', {'Train': epoch_train_accuracy, 'Evaluate': eval_accuracy}, epoch)

                print(f'Epoch {epoch + 1} / {self.epoch_size}, Evaluate Accuracy: {eval_accuracy}')

            self.scheduler.step(eval_loss)

    def print_train_test_accuracy(self):
        train_accuracy, _ = self.evaluate_model(is_valid=False, dataset='train')
        test_accuracy, _ = self.evaluate_model(is_valid=False, dataset='test')

        print(self.model_name)
        print(f'Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')
