import os
import numpy as np
import config
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from albumentations.pytorch import ToTensorV2

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        transformed_data = config.both_transforms(image=image)

        high_res = config.highres_transform(image=image)
        high_res = transformed_data["image"]

        low_res = config.lowres_transform(image=image)["image"]
        #low_res = transformed_data["image"]
        low_res = low_res.float()
        high_res = high_res.float()
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="./dataset/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print("=====================")
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()