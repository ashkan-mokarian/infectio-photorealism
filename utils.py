import os

import torch
from torchvision.utils import save_image

import config

def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint to: ", filename)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint from: ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_some_examples(gen, val_loader, epoch, fileroot):
    x, y = next(iter(val_loader))
    x, y = x.to(config.device), y.to(config.device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, os.path.join(fileroot, f"y_gen_{epoch}.png"))
        save_image(x * 0.5 + 0.5, os.path.join(fileroot, f"x_{epoch}.png"))
        save_image(y * 0.5 + 0.5, os.path.join(fileroot,f"y_gt_{epoch}.png"))
    gen.train()