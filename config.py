import torch
import os

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
TRAIN_DIR = "./dataset/M061/train"
VAL_DIR = "./dataset/M061/val"
LEARNING_RATE = 2e-4
BETA1 = 0.5  # From the paper
BATCH_SIZE = 16
NUM_WORKERS = 4
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
# False means train from scratch, with number means the checkpoint to load
LOAD_MODEL_EPOCH = False
NAME = "baseline"
OUTPUT_DIR = "./output/"
LOG_FREQ = 50
