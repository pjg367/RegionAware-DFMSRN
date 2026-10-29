from pathlib import Path
import torch

# Replace every SET_* placeholder before running training or inference.

# Set the MSRB root containing training/ and test/.
MSRB_ROOT = Path("SET_MSRB_ROOT")
TRAIN_TASK1_INPUT_DIR = MSRB_ROOT / 'training' / 'MSR_Task1'
TRAIN_TASK1_MASK_DIR = MSRB_ROOT / 'training' / 'MSR_Task1_binarymask'
TRAIN_TASK2_INPUT_DIR = MSRB_ROOT / 'training' / 'MSR_Task2'
TRAIN_TASK2_MASK_DIR = MSRB_ROOT / 'training' / 'MSR_Task2_binarymask'

# Set the training ground-truth image directory.
TRAIN_GT_DIR = Path("SET_TRAIN_GT_DIR")
VAL_TASK1_INPUT_DIR = MSRB_ROOT / 'test' / 'MSR_Task1'
VAL_TASK1_MASK_DIR = MSRB_ROOT / 'test' / 'MSR_Task1_binarymask'
VAL_TASK2_INPUT_DIR = MSRB_ROOT / 'test' / 'MSR_Task2'
VAL_TASK2_MASK_DIR = MSRB_ROOT / 'test' / 'MSR_Task2_binarymask'

# Set the test ground-truth image directory.
VAL_GT_DIR = Path("SET_TEST_GT_DIR")

# Set the proposed-model checkpoint output directory.
CHECKPOINT_DIR = Path("SET_CHECKPOINT_DIR")

# Set the proposed-model preview output directory.
PREVIEW_DIR = Path("SET_PREVIEW_DIR")
BATCH_SIZE = 4
EPOCHS = 80
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
LR_STEP_SIZE = 5
LR_GAMMA = 0.5
NUM_WORKERS = 4
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESUME_TRAINING = True
RESUME_CHECKPOINT = CHECKPOINT_DIR / 'dfmsrn_last.pth'
BEST_CHECKPOINT = CHECKPOINT_DIR / 'dfmsrn_best.pth'
PHI = 0.15
STAGE1_WEIGHT = PHI
STAGE2_WEIGHT = 1.0 - PHI
IMAGE_LOSS_WEIGHT = 1.0
FOURIER_LOSS_WEIGHT = 1.0
SNOW_LOSS_WEIGHT = 3.0
BACKGROUND_LOSS_WEIGHT = 2.0
CHARBONNIER_EPS = 0.001
PERCEPTUAL_WEIGHT = 0.1
STAGE2_SNOW_LOSS_WEIGHT = 2.0
STAGE2_BACKGROUND_LOSS_WEIGHT = 5.0
MASK_DILATION_KERNEL = 3
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# Set separate baseline output directories.
ORIGINAL_CHECKPOINT_DIR = Path("SET_ORIGINAL_CHECKPOINT_DIR")
ORIGINAL_PREVIEW_DIR = Path("SET_ORIGINAL_PREVIEW_DIR")

# Set the removal-model checkpoint and inference image directories.
CHECKPOINT_PATH = Path("SET_REMOVAL_CHECKPOINT_PATH")
INPUT_DIR = Path("SET_INPUT_DIR")
OUTPUT_DIR = Path("SET_OUTPUT_DIR")
SAVE_XMID = True
