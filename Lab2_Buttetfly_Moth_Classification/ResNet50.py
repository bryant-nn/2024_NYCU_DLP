import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils import data
from dataloader import BufferflyMothLoader

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride=1, downsample=False):
        super(Bottleneck, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels * self.expansion),
        )

        # Initialize the weights of the convolution layers
        # for m in self.block.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

        self.shortcut = nn.Sequential()

        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * self.expansion),
            )

            # Initialize the weights of the convolution layers
            # for m in self.shortcut.modules():
            #     if isinstance(m, nn.Conv2d):
            #         init.kaiming_normal_(m.weight)
            #         if m.bias is not None:
            #             init.constant_(m.bias, 0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print('x', x.shape)
        out = self.block(x)
        # print('out', out.shape)
        out += self.shortcut(x)
        # print('out', out.shape)
        out = self.relu(out)
        return out
    
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.inplane = 64
        self.num_class = num_classes
        # stage 0
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # initialize the weights of the convolution layers
        # for m in self.layer0.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

        # stage 1
        self.layer1 = nn.Sequential(
            self._make_layer(64, 64, Bottleneck, 3, stride=1)
        )

        # stage 2
        self.layer2 = nn.Sequential(
            self._make_layer(256, 128, Bottleneck, 4, stride=2)
        )

        # stage 3
        self.layer3 = nn.Sequential(
            self._make_layer(512, 256, Bottleneck, 6, stride=2)
        )

        # stage 4
        self.layer4 = nn.Sequential(
            self._make_layer(1024, 512, Bottleneck, 3, stride=2)
        )

        self.fc = nn.Linear(2048, num_classes)
        # Initialize the weights of the fully connected layers
        init.kaiming_normal_(self.fc.weight)
    
    def _make_layer(self, in_channels, channels, block, num_blocks, stride=1):
        layers = []
        layers.append(block(in_channels, channels, stride, downsample=True))
        # self.inplane = channels

        for _ in range(1, num_blocks):
            layers.append(block(channels * 4, channels, 1))
            # self.inplane = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer0(x)
        # print('1', x.shape)
        x = self.layer1(x)
        # print('2', x.shape)
        x = self.layer2(x)
        # print('3', x.shape)
        x = self.layer3(x)
        # print('4', x.shape)
        x = self.layer4(x)
        # print('5', x.shape)

        x = nn.AvgPool2d(7)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
# if __name__ == "__main__":
#     train_data = BufferflyMothLoader('./dataset/', 'train')
#     train_loader = data.DataLoader(
#         dataset=train_data, batch_size=32, shuffle=True)
#     model = ResNet50(100)
#     # print(model)

#     for img, label in train_loader:
#         # print(img.shape)
#         model.train()
#         output = model(img)
#         # print(label)
#         break

    # print(model)