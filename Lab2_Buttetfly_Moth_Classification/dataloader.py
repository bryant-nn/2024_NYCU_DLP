import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('./dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        df = pd.read_csv('./dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    else:
        df = pd.read_csv('./dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or validing or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        # print("> Found %d images..." % (len(self.img_name)))  

        # 定義圖像轉換
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 調整圖像大小為 224x224
            transforms.RandomHorizontalFlip(),  # 隨機對圖像執行水平翻轉
            transforms.RandomRotation(15),  # 隨機旋轉圖像
            transforms.RandomCrop(224, padding=8),  # 隨機裁剪圖像
            transforms.ToTensor(),  # 轉換圖像為 tensor，並將像素值範圍調整到 [0, 1]
        ])

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        img = Image.open(self.root + self.img_name[index])
        label = self.label[index]
        # print(img.size)
        

        # img = img.convert('RGB')
        if self.mode == 'train':
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        # print(img.shape)
        # normalize
        
        return img, label

# if __name__ == '__main__':
#     train_data = BufferflyMothLoader('./dataset/', 'train')
#     train_loader = data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
#     for img, label in train_loader:
#         # print(img)
#         print(label)
#         break
