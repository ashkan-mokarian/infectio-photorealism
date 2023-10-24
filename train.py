import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


from model.model import Discriminator, Generator
from data.plaque_data import InfSpots2Plaque_Data
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


def eval(netG, netD, val_dl, L1_Loss, BCE_Loss):
    Disc_loss_val = []
    Gen_loss_val = []

    netG.eval()
    netD.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(val_dl)):
            x = x.to(config.device)
            y = y.to(config.device)

            # Discriminator
            y_fake = netG(x)
            D_real = netD(x, y)
            D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
            D_fake = netD(x, y_fake.detach())
            D_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            Disc_loss_val.append(D_loss.item())

            # Generator
            D_fake = netD(x, y_fake)
            G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
            L1 = L1_Loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

            Gen_loss_val.append(G_loss.item())

    netG.train()
    netD.train()
    return Disc_loss_val, Gen_loss_val


def main():
    SAVE_ROOT = os.path.join(config.OUTPUT_DIR, config.NAME)
    CKPT_ROOT = os.path.join(SAVE_ROOT, "ckpts")
    SAVE_EXAMPLE_IMAGE_ROOT = os.path.join(SAVE_ROOT, "examples")
    CHECKPOINT_DISC = "disc-{0}.pth.tar"
    CHECKPOINT_GEN = "gen-{0}.pth.tar"

    netD = Discriminator(in_channels=1).to(config.device)
    netG = Generator(in_channels=1).to(config.device)
    optimD = torch.optim.Adam(
        netD.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999)
    )
    optimG = torch.optim.Adam(
        netG.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999)
    )
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    train_dataset = InfSpots2Plaque_Data(config.TRAIN_DIR, train=True)
    train_dl = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_dataset = InfSpots2Plaque_Data(config.VAL_DIR, train=False)
    val_dl = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    os.makedirs(CKPT_ROOT, exist_ok=True)
    os.makedirs(SAVE_EXAMPLE_IMAGE_ROOT, exist_ok=True)
    # Save config file for bookkeeping
    shutil.copyfile(config.__file__, os.path.join(SAVE_ROOT, "config.py"))

    writer = SummaryWriter(comment=config.NAME)

    starting_epoch = 1
    if config.LOAD_MODEL_EPOCH:
        starting_epoch = config.LOAD_MODEL_EPOCH + 1
        load_checkpoint(
            os.path.join(
                CKPT_ROOT,
                CHECKPOINT_GEN.format(config.LOAD_MODEL_EPOCH),
            ),
            netG,
            optimG,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(
                CKPT_ROOT,
                CHECKPOINT_DISC.format(config.LOAD_MODEL_EPOCH),
            ),
            netD,
            optimD,
            config.LEARNING_RATE,
        )
        print(f"Loaded model from epoch {config.LOAD_MODEL_EPOCH}")

    for epoch in range(starting_epoch, config.NUM_EPOCHS + 1):
        disc_loss, gen_loss = train(
            netG, netD, train_dl, optimG, optimD, L1_Loss, BCE_Loss
        )
        print(
            f"Finished epoch {epoch}/{config.NUM_EPOCHS}: "
            f"average disc-loss:{sum(disc_loss)/float(len(disc_loss)):.6f} "
            f"average gen-loss:{sum(gen_loss)/float(len(gen_loss)):.6f}"
        )
        global_step = epoch * len(disc_loss)
        writer.add_scalar(
            "Loss/Disc/Train", sum(disc_loss) / float(len(disc_loss)), global_step
        )
        writer.add_scalar(
            "Loss/Gen/Train", sum(gen_loss) / float(len(gen_loss)), global_step
        )

        # Let's validate every 5 epochs.
        if epoch % 5 == 0:
            disc_loss, gen_loss = eval(netG, netD, val_dl, L1_Loss, BCE_Loss)
            writer.add_scalar(
                "Loss/Disc/Val", sum(disc_loss) / float(len(disc_loss)), global_step
            )
            writer.add_scalar(
                "Loss/Gen/Val", sum(gen_loss) / float(len(gen_loss)), global_step
            )
            print(
                f"Finished VALIDATION for epoch {epoch}/{config.NUM_EPOCHS}: "
                f"average disc-loss:{sum(disc_loss)/float(len(disc_loss)):.6f} "
                f"average gen-loss:{sum(gen_loss)/float(len(gen_loss)):.6f}"
            )

        if (epoch) % config.LOG_FREQ == 0:
            save_checkpoint(
                netG,
                optimG,
                os.path.join(CKPT_ROOT, CHECKPOINT_GEN.format(epoch)),
            )
            save_checkpoint(
                netD,
                optimD,
                os.path.join(CKPT_ROOT, CHECKPOINT_DISC.format(epoch)),
            )
            y_gen = save_some_examples(
                netG, train_dl, epoch, fileroot=SAVE_EXAMPLE_IMAGE_ROOT
            )
            writer.add_image(
                "Train", make_grid(y_gen, nrow=4, normalize=True), global_step
            )
            writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
