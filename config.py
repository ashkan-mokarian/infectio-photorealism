import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
TRAIN_DIR = "./dataset/maps/train"
VAL_DIR = "./dataset/maps/val"
LEARNING_RATE = 2e-4
BETA1 = 0.5
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 4
LOAD_MODEL = False
SAVE_MODEL_ROOT = "./output/maps/ckpts/"
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
SAVE_EXAMPLE_IMAGE_ROOT = "./output/maps/examples"