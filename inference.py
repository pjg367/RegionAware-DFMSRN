from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import (
    CHECKPOINT_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    SAVE_XMID,
    DEVICE,
    IMAGE_EXTENSIONS,
)
from models import DFMSRN

def list_images(folder):

    folder = Path(folder)

    if not folder.exists():

        raise FileNotFoundError(f'\nInput directory does not exist.\n{folder}')

    images = sorted([
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ])

    if len(images) == 0:

        raise RuntimeError(f'\nNo input images found.\n{folder}')

    return images

def load_image(path):

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:

        raise RuntimeError(f'\nUnable to read the image.\n{path}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0

    original_height = image.shape[0]

    original_width = image.shape[1]

    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(image).float()

    tensor = tensor.unsqueeze(0)

    return (tensor, original_height, original_width)

def pad_to_multiple_of_16(image):

    (h, w) = image.shape[-2:]

    target_h = (h + 15) // 16 * 16

    target_w = (w + 15) // 16 * 16

    pad_h = target_h - h

    pad_w = target_w - w

    if (
        pad_h == 0
        and pad_w == 0
    ):

        return (image, 0, 0)

    # Reflection-pad the right and bottom edges.

    padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')

    return (padded, pad_h, pad_w)

def remove_padding(image, original_height, original_width):

    return image[:, :, :original_height, :original_width]

def save_tensor_image(tensor, output_path):

    # Clamp to [0, 1] only when saving.

    tensor = torch.clamp(tensor.detach(), 0.0, 1.0)

    if tensor.ndim == 4:

        tensor = tensor[0]

    image = tensor.cpu().numpy()

    image = np.transpose(image, (1, 2, 0))

    image = (image * 255.0).round().astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), image)

    if not success:

        raise RuntimeError(f'\nFailed to save the image\n{output_path}')

def load_model():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(f'\nCheckpoint does not exist.\n{CHECKPOINT_PATH}')

    model = DFMSRN().to(DEVICE)

    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)

    # Accept a training checkpoint or a raw model state dictionary.

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint['model_state_dict']

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    print()
    print("=" * 70)
    print('DF-MSRN Checkpoint Loaded')
    print("=" * 70)

    print(f'Checkpoint : {CHECKPOINT_PATH}')

    if isinstance(
        checkpoint,
        dict,
    ):

        if "epoch" in checkpoint:

            print(f"Epoch      : {checkpoint['epoch']}")

        if "val_total_loss" in checkpoint:

            print(f"Val Loss   : {checkpoint['val_total_loss']:.6f}")

        if "val_mid_psnr" in checkpoint:

            print(f"X_mid PSNR : {checkpoint['val_mid_psnr']:.3f} dB")

        if "val_fin_psnr" in checkpoint:

            print(f"X_fin PSNR : {checkpoint['val_fin_psnr']:.3f} dB")

        if "phi" in checkpoint:

            print(f"Phi        : {checkpoint['phi']}")

    print("=" * 70)
    print()

    return model

@torch.inference_mode()
def inference_one_image(model, input_path):

    (image, original_height, original_width) = load_image(input_path)

    image = image.to(DEVICE)

    (padded_image, pad_h, pad_w) = pad_to_multiple_of_16(image)

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    start_time = time.time()

    outputs = model(padded_image)

    x_mid = outputs['x_mid']

    x_fin = outputs['x_fin']

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    x_mid = remove_padding(x_mid, original_height, original_width)

    x_fin = remove_padding(x_fin, original_height, original_width)

    stats = {
        "height":
            original_height,
        "width":
            original_width,
        "pad_h":
            pad_h,
        "pad_w":
            pad_w,
        "input_min":
            image.min().item(),
        "input_max":
            image.max().item(),
        "x_mid_min":
            x_mid.min().item(),
        "x_mid_max":
            x_mid.max().item(),
        "x_fin_min":
            x_fin.min().item(),
        "x_fin_max":
            x_fin.max().item(),
        "time":
            elapsed,
    }

    return (x_mid, x_fin, stats)

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_XMID:

        xmid_dir = OUTPUT_DIR / 'x_mid'

        xmid_dir.mkdir(parents=True, exist_ok=True)

    xfin_dir = OUTPUT_DIR / 'x_fin'

    xfin_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()

    image_paths = list_images(INPUT_DIR)

    print()
    print("=" * 70)
    print('DF-MSRN Inference')
    print("=" * 70)

    print(f'Device      : {DEVICE}')

    if torch.cuda.is_available():

        print(f'GPU         : {torch.cuda.get_device_name(0)}')

    print(f'Input Dir   : {INPUT_DIR}')

    print(f'Output Dir  : {OUTPUT_DIR}')

    print(f'Images      : {len(image_paths)}')

    print(f'Save X_mid  : {SAVE_XMID}')

    print("=" * 70)
    print()

    total_time = 0.0

    for input_path in tqdm(
        image_paths,
        desc="DF-MSRN Inference",
    ):

        (x_mid, x_fin, stats) = inference_one_image(model, input_path)

        # Save PNG outputs to avoid JPEG recompression artifacts.

        output_name = input_path.stem + '.png'

        if SAVE_XMID:

            save_tensor_image(x_mid, xmid_dir / output_name)

        save_tensor_image(x_fin, xfin_dir / output_name)

        total_time += stats['time']

        print()
        print("-" * 70)

        print(f'File       : {input_path.name}')

        print(f"Resolution : {stats['width']}x{stats['height']}")

        print(f"Padding    : H +{stats['pad_h']}, W +{stats['pad_w']}")

        print(f"X_mid range: {stats['x_mid_min']:.4f} ~ {stats['x_mid_max']:.4f}")

        print(f"X_fin range: {stats['x_fin_min']:.4f} ~ {stats['x_fin_max']:.4f}")

        print(f"Time       : {stats['time']:.4f} sec")

        print("-" * 70)

    average_time = total_time / len(image_paths)

    print()
    print("=" * 70)
    print('Inference Finished')
    print("=" * 70)

    print(f'Images       : {len(image_paths)}')

    print(f'Total Time   : {total_time:.3f} sec')

    print(f'Average Time : {average_time:.3f} sec/image')

    if average_time > 0:

        print(f'Approx FPS   : {1.0 / average_time:.2f}')

    print()

    if SAVE_XMID:

        print(f'X_mid Output : {xmid_dir}')

    print(f'X_fin Output : {xfin_dir}')

    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
