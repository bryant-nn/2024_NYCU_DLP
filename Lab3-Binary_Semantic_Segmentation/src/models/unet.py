import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from .. import oxford_pet
# from oxford_pet import load_dataset
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Initialize the weights of the convolution layers
        for m in self.block.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)
    
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoders.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoders.append(
                    nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(
                    DoubleConv(feature * 2, feature),
            )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
        skip_connections = []
        
        # print('Encoder')
        for encoder in self.encoders:
            # print(x.shape)
            x = encoder(x)
            # print(x.shape)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.bottleneck(x)
        

        skip_connections.reverse()

        # print('Decoder')
        for i in range(0, len(self.decoders) , 2):
            # print(x.shape)
            x = self.decoders[i](x)
            # print(x.shape)
            # print(skip_connections[i // 2].shape)
            # print('===')

            # 調整解碼器輸出的空間尺寸，以與編碼器輸出的空間尺寸匹配
            target_height = skip_connections[i // 2].shape[2]
            target_width = skip_connections[i // 2].shape[3]
            x = F.interpolate(x, size=(target_height, target_width), mode='bilinear', align_corners=True)
            x = torch.cat((x, skip_connections[i // 2]), dim=1) # Batch_size, Channel, Height, Width
            # print(x.shape)
            x = self.decoders[i + 1](x)

        return self.final_conv(x)
    
if __name__ == "__main__":
    model = Unet(3, 1).to('cuda')
    # training_data = load_dataset('../dataset/oxford-iiit-pet', 'train')
    # training_loader = data.DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
    # for i, datas in enumerate(training_loader):
    #     x, y = datas['image'], datas['mask']
    #     print(x.shape)
    #     print(model(x).shape)
    #     break
    x = torch.randn((1, 3, 520, 333)).to('cuda')
    print(model(x).shape)