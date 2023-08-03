import os
import tifffile
import pandas as pd
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader

from model.model import Generator
from data.plaque_data import InfSpots2Plaque_Data
import config


class InfSpotsCSV_Data(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.df = pd.read_csv(csv_file, low_memory=False)

    def __len__(self):
        return int(pd.to_numeric(self.df["FRAME"], errors="coerce").max()) + 1

    def __getitem__(self, index):
        df_frame = self.df[self.df["FRAME"] == str(index)]
        pos_x = df_frame["POSITION_X"].astype(float).values
        pos_y = df_frame["POSITION_Y"].astype(float).values
        pos_x /= 3.1746
        pos_y /= 3.1746
        pos = np.concatenate(
            (pos_x.reshape(-1, 1).astype(int), pos_y.reshape(-1, 1).astype(int)), axis=1
        )
        spots = torch.zeros((512, 672), dtype=torch.float32)
        spots[pos[:, 1], pos[:, 0]] = 1
        return spots.unsqueeze(0)


def predict(netg, csv_file, output_tif):
    """Saves prediction results as tif file with multiple frames based on a csv
    input file containing x, y positions and frame numbers as columns.

    Args:
        netg (torch.nn.Module): Generator model.
        csv_file (str): Path to csv file containing x, y positions and frame
            numbers.
        output_tif (str): Output tif path to be saved.
    """
    dataset = InfSpotsCSV_Data(csv_file)
    dataloader = DataLoader(dataset, batch_size=13, shuffle=False)
    netg.eval()
    all_frames = []
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            x = x.to(config.device)
            pred = netg(x)
            pred = pred.squeeze(1).cpu().numpy()
            pred = (pred * 255).astype("uint8")
            all_frames.append(pred)

    tifffile.imwrite(output_tif, np.concatenate(all_frames, axis=0))


if __name__ == "__main__":
    checkpoint_file = "./output/plaques/ckpts/gen.pth.tar"
    input_csv = "./dataset/M061/val/M061_14.csv"
    gt_tif = "./dataset/M061/val/M061_14.tif"
    output_tif = "./output/plaques/evaluate/M061_14.tif"
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    netg = Generator(in_channels=1).to(config.device)
    netg.load_state_dict(
        torch.load(checkpoint_file, map_location=config.device)["state_dict"]
    )
    predict(netg, input_csv, output_tif)

    # Concatenate it with gt tif and compare their difference too
    gt_img = tifffile.imread(gt_tif)
    pred_img = tifffile.imread(output_tif)
    diff = np.abs(np.subtract(gt_img.astype("int32"), pred_img.astype("int32")))
    evaluate_img = np.concatenate(
        [gt_img, pred_img, diff.astype("uint8")],
        axis=1,
    )
    tifffile.imwrite(output_tif, evaluate_img)
