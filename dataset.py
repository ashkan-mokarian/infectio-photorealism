import os
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import random # FIXME set random seed


def both_transform(sample):
    im1, im2 = sample
    im1, im2 = TF.to_tensor(im1), TF.to_tensor(im2)
    if 0.5 > random.random():
        im1, im2 = TF.hflip(im1), TF.hflip(im2)
    angle = random.randrange(-30, +30)
    im1, im2 = TF.rotate(im1, angle), TF.rotate(im2, angle)
    transform = T.Compose([
        T.Resize((256, 256))
    ])
    return (transform(im1), transform(im2))

class Sattelite2Map_Data(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.n_samples = os.listdir(self.root)
    
    def __len__(self):
        return len(self.n_samples)
    
    def __getitem__(self, index):
        image_name = self.n_samples[index]
        image_path = os.path.join(self.root, image_name)
        image = np.asarray(Image.open(image_path).convert('RGB'))
        height, width, _ = image.shape
        width_cutoff = width // 2
        satellite_image = image[:, :width_cutoff, :]
        map_image = image[:, width_cutoff:, :]

        # Transforms
        transformed_inputs = both_transform((satellite_image, map_image))
        return transformed_inputs

if __name__ == '__main__':
    dataset = Sattelite2Map_Data('./dataset/maps/train')
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print("X Shape :-", x.shape)
        print("Y Shape :-", y.shape)
        save_image(x, 'satellite.png')
        save_image(y, 'map.png')
        break