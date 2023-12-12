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
    # angle = random.randrange(-30, +30)
    # spots, image = TF.rotate(spots, angle), TF.rotate(image, angle)
    # image = TF.normalize(image, mean=0.042, std=0.0156)
    spots, image = TF.resize(spots, size=(512, 512), antialias=True), TF.resize(
        image, size=(512, 512), antialias=True
    )

    return (spots, image)


class InfSpots2Plaque_Data(Dataset):
    def __init__(self, root, train=True) -> None:
        super().__init__()
        self.root = root
        self.n_samples = [
            f.split(".")[0] for f in os.listdir(self.root) if f.endswith(".csv")
        ]
        # Sort samples based on their number. Each 15 partition belongs to a
        # different strain. 1-15(WR), 16-30(dVGF), 31-45(dF11), 46-60(dVGF/dF11)
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
        df = df.drop(index=df.index[:3])
        df["FRAME"] = df["FRAME"].astype(int)
        df_frame = df[df["FRAME"] == n_frame]
        pos_x = df_frame["POSITION_X"].astype(float).values
        pos_y = df_frame["POSITION_Y"].astype(float).values
        # positions in csv (trackmate) are given in micro meters and not pixels
        # sampling of microscope is 3.1746 x 3.1746 um per pixel
        pos_x /= 3.1746
        pos_y /= 3.1746
        pos = np.concatenate(
            (pos_x.reshape(-1, 1).astype(int), pos_y.reshape(-1, 1).astype(int)), axis=1
        )

        # For each spot, we also want to know its time being already infected.
        # We think this is a valuable information that defines brightness of spots.
        track_id = df_frame["TRACK_ID"].values
        min_frame_per_track = df.groupby("TRACK_ID")["FRAME"].min().reset_index()
        min_frames_list = []
        for i in track_id:
            min_frame = min_frame_per_track[min_frame_per_track["TRACK_ID"] == i][
                "FRAME"
            ].values
            if len(min_frame) == 1:
                min_frames_list.append(int(min_frame[0]))
            else:
                raise ValueError("More than one min frame for a track id")
        inf_frame_time = np.array(
            [n_frame - min_frame for min_frame in min_frames_list]
        )
        spot_values = np.floor(inf_frame_time / (self.n_frames - 1) * 255)

        # Create a numpy array with values for spots defined by how long they
        # have been infected
        spots = np.zeros_like(image)
        spots[pos[:, 1], pos[:, 0]] = spot_values

        # Transforms
        inputs = (spots, image)
        return transform(inputs, self.train)


if __name__ == "__main__":
    dataset = InfSpots2Plaque_Data("./dataset/M061/train")
    print("Dataset Length :-", len(dataset))
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for x, y in loader:
        print("X Shape :-", x.shape)
        print("Y Shape :-", y.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        break
