import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
PIN_MEMORY = True
VAL_IMG_DIR = 'test-set'
VAL_MASK_DIR = 'test-set-masks'
