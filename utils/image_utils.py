import cv2
import numpy as np
import torch

def tensor_to_rgb_uint8(tensor):

    tensor = torch.clamp(tensor.detach(), 0.0, 1.0)

    if tensor.ndim == 4:

        tensor = tensor[0]

    image = tensor.cpu().numpy()

    image = np.transpose(image, (1, 2, 0))

    image = (image * 255.0).round().astype(np.uint8)

    return image

def save_tensor_image(tensor, output_path):

    image = tensor_to_rgb_uint8(tensor)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), image)

    if not success:

        raise RuntimeError(f'Failed to save the image:\n{output_path}')

def save_mask(mask, output_path):

    mask = torch.clamp(mask.detach(), 0.0, 1.0)

    if mask.ndim == 4:

        mask = mask[0]

    if mask.ndim == 3:

        mask = mask[0]

    image = (mask.cpu().numpy() * 255.0).round().astype(np.uint8)

    cv2.imwrite(str(output_path), image)
