import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import IMAGE_EXTENSIONS

def list_images(folder):

    folder = Path(folder)

    if not folder.exists():

        raise FileNotFoundError(f'\nDirectory does not exist.\n{folder}\n')

    images = sorted([
        path
        for path
        in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])

    if len(images) == 0:

        raise RuntimeError(f'\nNo images found.\n{folder}\n')

    return images

def silent_imread(path, flag):

    old_stderr_fd = os.dup(2)

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        try:
            os.dup2(devnull_fd, 2)

            image = cv2.imread(str(path), flag)

        finally:
            os.close(devnull_fd)

    finally:
        os.dup2(old_stderr_fd, 2)

        os.close(old_stderr_fd)

    return image

class PairedMarineSnowDataset(Dataset):

    def __init__(self, input_dir, gt_dir, mask_dir, training=True, dataset_name=None):

        super().__init__()

        self.input_dir = Path(input_dir)

        self.gt_dir = Path(gt_dir)

        self.mask_dir = Path(mask_dir)

        self.training = training

        self.dataset_name = (
            dataset_name
            if dataset_name
            is not None
            else (
                "TRAIN"
                if training
                else "VALIDATION"
            )
        )

        input_paths = list_images(self.input_dir)

        gt_paths = list_images(self.gt_dir)

        mask_paths = list_images(self.mask_dir)

        gt_dict = {path.name: path for path in gt_paths}

        mask_dict = {path.name: path for path in mask_paths}

        self.samples = []

        missing_gt = []

        missing_mask = []

        for input_path in input_paths:

            filename = input_path.name

            gt_path = gt_dict.get(filename)

            mask_path = mask_dict.get(filename)

            if gt_path is None:

                missing_gt.append(filename)

                continue

            if mask_path is None:

                missing_mask.append(filename)

                continue

            self.samples.append((input_path, gt_path, mask_path))

        if len(self.samples) == 0:

            raise RuntimeError(
                "\nNo valid "
                "Input / GT / Mask pairs found.\n"
                f"Input : {self.input_dir}\n"
                f"GT    : {self.gt_dir}\n"
                f"Mask  : {self.mask_dir}\n"
            )

        print()

        print("=" * 70)

        print(self.dataset_name)

        print("=" * 70)

        print(f'Input images : {len(input_paths)}')

        print(f'GT images    : {len(gt_paths)}')

        print(f'Mask images  : {len(mask_paths)}')

        print(f'Valid pairs  : {len(self.samples)}')

        print(f'Missing GT   : {len(missing_gt)}')

        print(f'Missing Mask : {len(missing_mask)}')

        print("=" * 70)

        print()

    def __len__(self):

        return len(self.samples)

    @staticmethod

    def load_image(path):

        image = silent_imread(path, cv2.IMREAD_COLOR)

        if image is None:

            raise RuntimeError(f'Unable to read the image:\n{path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        return image

    @staticmethod

    def load_mask(path):

        mask = silent_imread(path, cv2.IMREAD_GRAYSCALE)

        if mask is None:

            raise RuntimeError(f'Unable to read the mask:\n{path}')

        mask = mask.astype(np.float32) / 255.0

        # background = 0

        # marine snow = 1

        mask = (mask > 0.5).astype(np.float32)

        return mask

    def __getitem__(self, index):

        (input_path, gt_path, mask_path) = self.samples[index]

        input_image = self.load_image(input_path)

        gt_image = self.load_image(gt_path)

        mask = self.load_mask(mask_path)

        if (
            input_image.shape
            != gt_image.shape
        ):

            raise RuntimeError(
                "\nInput / GT shape mismatch\n"
                f"Input : {input_path}\n"
                f"Shape : {input_image.shape}\n"
                f"GT    : {gt_path}\n"
                f"Shape : {gt_image.shape}"
            )

        if (
            input_image.shape[:2]
            != mask.shape[:2]
        ):

            raise RuntimeError(
                "\nInput / Mask shape mismatch\n"
                f"Input : {input_path}\n"
                f"Shape : {input_image.shape[:2]}\n"
                f"Mask  : {mask_path}\n"
                f"Shape : {mask.shape[:2]}"
            )

        # Full Image

        # No crop

        input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).float()

        gt_tensor = torch.from_numpy(gt_image.transpose(2, 0, 1)).float()

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return (input_tensor, gt_tensor, mask_tensor, input_path.name)

def original_silent_imread(path, flags):

    stderr_fd = 2

    saved_stderr_fd = os.dup(stderr_fd)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:

        os.dup2(devnull_fd, stderr_fd)

        image = cv2.imread(str(path), flags)

    finally:

        os.dup2(saved_stderr_fd, stderr_fd)

        os.close(saved_stderr_fd)

        os.close(devnull_fd)

    return image

class OriginalMarineSnowDataset(Dataset):

    def __init__(self, input_dir, gt_dir, training=True, dataset_name=None):

        super().__init__()

        self.input_dir = Path(input_dir)

        self.gt_dir = Path(gt_dir)

        self.training = training

        self.dataset_name = (
            dataset_name
            if dataset_name is not None
            else (
                "TRAIN"
                if training
                else "VALIDATION"
            )
        )

        input_paths = list_images(self.input_dir)

        gt_paths = list_images(self.gt_dir)

        gt_dict = {path.name: path for path in gt_paths}

        self.samples = []

        missing_gt = []

        for input_path in input_paths:

            filename = input_path.name

            gt_path = gt_dict.get(filename)

            if gt_path is None:

                missing_gt.append(filename)

                continue

            self.samples.append((input_path, gt_path))

        if len(self.samples) == 0:

            raise RuntimeError(
                "\nNo valid "
                "Input / GT pairs found.\n"
                f"Input : {self.input_dir}\n"
                f"GT    : {self.gt_dir}\n"
            )

        print()

        print("=" * 70)

        print(self.dataset_name)

        print("=" * 70)

        print(f'Input images : {len(input_paths)}')

        print(f'GT images    : {len(gt_paths)}')

        print(f'Valid pairs  : {len(self.samples)}')

        print(f'Missing GT   : {len(missing_gt)}')

        print("=" * 70)

        print()

    def __len__(self):

        return len(self.samples)

    @staticmethod

    def load_image(path):

        image = original_silent_imread(path, cv2.IMREAD_COLOR)

        if image is None:

            raise RuntimeError(f'Unable to read the image:\n{path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        return image

    def __getitem__(self, index):

        (input_path, gt_path) = self.samples[index]

        input_image = self.load_image(input_path)

        gt_image = self.load_image(gt_path)

        if (
            input_image.shape
            != gt_image.shape
        ):

            raise RuntimeError(
                "\nInput / GT shape mismatch\n"
                f"Input : {input_path}\n"
                f"Shape : {input_image.shape}\n"
                f"GT    : {gt_path}\n"
                f"Shape : {gt_image.shape}"
            )

        # Use full images without cropping.

        input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).float()

        gt_tensor = torch.from_numpy(gt_image.transpose(2, 0, 1)).float()

        return (input_tensor, gt_tensor, input_path.name)
