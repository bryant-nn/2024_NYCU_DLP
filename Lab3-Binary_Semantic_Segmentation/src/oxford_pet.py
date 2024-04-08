import os
import torch
import shutil
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from utils import dice_score

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        # print("sample", sample)
        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])
            sample["mask"] = self.transform(sample["mask"])
            sample["trimap"] = self.transform(sample["trimap"])

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    # implement the load dataset function here
    # if data_path does not exist, download the dataset
    if not os.path.exists(data_path):
        OxfordPetDataset.download(data_path)
    if mode == "train":
        # 定義圖像轉換
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 調整圖像大小為 224x224
            transforms.RandomHorizontalFlip(),  # 隨機對圖像執行水平翻轉
            transforms.RandomRotation(15),  # 隨機旋轉圖像
            # transforms.RandomCrop(256, padding=8),  # 隨機裁剪圖像
            transforms.ToTensor(),  # 轉換圖像為 tensor，並將像素值範圍調整到 [0, 1]
        ])
        data_loader = OxfordPetDataset(data_path, mode, transform)
        # print(data_loader[1])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 調整圖像大小為 224x224
            transforms.ToTensor(),  # 轉換圖像為 tensor，並將像素值範圍調整到 [0, 1]
        ])
        data_loader = OxfordPetDataset(data_path, mode, transform)
    return data_loader

if __name__ == "__main__":
    dataloader = load_dataset('../dataset/oxford-iiit-pet', 'train')
    # print(len(dataloader))
    for data_ in dataloader:
        sample = data_
        print((sample['image']).shape)
        

    # pred_mask_batch = torch.rand(32, 1, 256, 256, requires_grad=True)
    # print(pred_mask_batch.shape)

    # # pred_mask_batch = pred_mask_batch.sum()
    # # print(pred_mask_batch)

    # pred_mask_batch.item()
    # print(pred_mask_batch.shape)

