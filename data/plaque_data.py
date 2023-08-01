import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from tifffile import TiffFile
import numpy as np
import random
import pandas as pd


def transform(sample, train):
    spots, image = sample
    spots, image = TF.to_tensor(spots), TF.to_tensor(image)
    if train:
        if 0.5 > random.random():
            spots, image = TF.hflip(spots), TF.hflip(image)
        if 0.5 > random.random():
            spots, image = TF.vflip(spots), TF.vflip(image)
        angle = random.randrange(-30, +30)
        spots, image = TF.rotate(spots, angle), TF.rotate(image, angle)
    # image = TF.normalize(image, mean=0.042, std=0.0156)

    return (spots, image)


class InfSpots2Plaque_Data(Dataset):
    def __init__(self, root, train=True) -> None:
        super().__init__()
        self.root = root
        self.n_samples = [
            f.split(".")[0] for f in os.listdir(self.root) if f.endswith(".csv")
        ]
        # Sort samples based on their number. Each 15 partition belongs to a
        # different strain
        self.n_samples.sort(key=lambda x: int(x.split("_")[1]))
        for f in self.n_samples:
            assert os.path.exists(
                os.path.join(self.root, f + ".tif")
            ), f"tif file for {f} not found"
        # All tif files have the same 169 frames.
        self.n_frames = 169
        self.train = train

    def __len__(self):
        return len(self.n_samples) * self.n_frames

    def __getitem__(self, index):
        n_sample = index // self.n_frames
        n_frame = index % self.n_frames
        sample_name = self.n_samples[n_sample]
        image_path = os.path.join(self.root, sample_name + ".tif")
        csv_path = os.path.join(self.root, sample_name + ".csv")
        # Read image frame
        with TiffFile(image_path) as tif:
            image = tif.pages[n_frame].asarray()
        # Read x, y pos of spots for the corresponding frame
        df = pd.read_csv(os.path.join(csv_path), low_memory=False)
        df_frame = df[df["FRAME"] == str(n_frame)]
        pos_x = df_frame["POSITION_X"].astype(float).values
        pos_y = df_frame["POSITION_Y"].astype(float).values
        # positions in csv (trackmate) are given in micro meters and not pixels
        # sampling of microscope is 3.1746 x 3.1746 um per pixel
        pos_x /= 3.1746
        pos_y /= 3.1746
        pos = np.concatenate(
            (pos_x.reshape(-1, 1).astype(int), pos_y.reshape(-1, 1).astype(int)), axis=1
        )
        # Create a numpy array with values for spots as one
        spots = np.zeros_like(image)
        spots[pos[:, 1], pos[:, 0]] = 255

        # Transforms
        inputs = (spots, image)
        return transform(inputs, self.train)


if __name__ == "__main__":
    dataset = InfSpots2Plaque_Data("./dataset/M061/train")
    print("Dataset Length :-", len(dataset))
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print("X Shape :-", x.shape)
        print("Y Shape :-", y.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        break
