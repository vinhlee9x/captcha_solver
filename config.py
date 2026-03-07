import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
CHAR2INT = {c: i + 1 for i, c in enumerate(CHARS)}
INT2CHAR = {i + 1: c for i, c in enumerate(CHARS)}

NUM_CLASSES = len(CHARS) + 1

MODEL_PATH = 'captcha_crnn_model.pth'
DATASET_DIR = 'my_dataset'
REAL_DATASET_DIR = 'real_dataset'
REAL_LABELED_DIR = 'real_dataset/labeled'