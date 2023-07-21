import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Discriminator, Generator
from dataset import Sattelite2Map_Data
from utils import save_checkpoint, load_checkpoint, save_some_examples
import config


def train(netG, netD, train_dl, optimG, optimD, L1_Loss, BCE_Loss):
    Disc_loss_train = []
    Gen_loss_train = []

    for idx, (x, y) in enumerate(tqdm(train_dl)):
        x = x.to(config.device)
        y = y.to(config.device)
        
        # Discriminator training
        y_fake = netG(x)
        D_real = netD(x, y)
        D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
        D_fake = netD(x, y_fake.detach())
        D_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        Disc_loss_train.append(D_loss.item())
        optimD.zero_grad()
        D_loss.backward()
        optimD.step()

        # Generator training
        D_fake = netD(x, y_fake)
        G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
        L1 = L1_Loss(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        Gen_loss_train.append(G_loss.item())
        optimG.zero_grad()
        G_loss.backward()
        optimG.step()

    return Disc_loss_train, Gen_loss_train



def main():
    netD = Discriminator(in_channels=3).to(config.device)
    netG = Generator(in_channels=3).to(config.device)
    optimD = torch.optim.Adam(netD.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    train_dataset = Sattelite2Map_Data(config.TRAIN_DIR, train=True)
    train_dl = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_dataset = Sattelite2Map_Data(config.VAL_DIR, train=False)
    val_dl = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True)
    
    os.makedirs(config.SAVE_MODEL_ROOT, exist_ok=True)
    os.makedirs(config.SAVE_EXAMPLE_IMAGE_ROOT, exist_ok=True)
    for epoch in range(config.NUM_EPOCHS):
        disc_loss, gen_loss = train(netG, netD, train_dl, optimG, optimD, L1_Loss, BCE_Loss)
        print(f"Finished epoch {epoch+1}/{config.NUM_EPOCHS}: "
              f"average disc-loss:{sum(disc_loss)/float(len(disc_loss)):.4f} "
              f"average gen-loss:{sum(gen_loss)/float(len(gen_loss)):.4f}")
        if (epoch + 1) % 5 == 0:
            save_checkpoint(netG, optimG, os.path.join(config.SAVE_MODEL_ROOT, config.CHECKPOINT_GEN))
            save_checkpoint(netD, optimD, os.path.join(config.SAVE_MODEL_ROOT, config.CHECKPOINT_DISC))
        if (epoch + 1) % 3 == 0:
            save_some_examples(netG, val_dl, epoch, fileroot=config.SAVE_EXAMPLE_IMAGE_ROOT)

if __name__ == "__main__":
    main()