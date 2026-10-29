# Marine Snow Removal with Region-Aware Training

Training and inference code for a two-stage marine snow removal network, including a baseline trained with the original DF-MSRN losses. Both training objectives use the same network architecture.

This repository contains the removal network only. Datasets, pretrained removal checkpoints, WGAN training code, generator and critic implementations, and synthetic marine snow generation scripts are not included.

## Network

Stage 1 restores the Fourier amplitude and reconstructs an intermediate image, `x_mid`, using the input phase. Stage 2 processes `x_mid` in the spatial and Fourier domains, fuses the features using high-frequency information from `x_mid` and the original input, and predicts a residual:

```text
Input image
    |
    v
Stage 1: Fourier amplitude restoration
    |
    v
x_mid
    |
    v
Stage 2: spatial/Fourier processing and HFE-based fusion
    |
    v
x_fin = x_mid + residual
```

The full model returns `x_mid`, `x_fin`, `predicted_amplitude`, `raw_amplitude`, `theta`, `f_spa`, and `f_fre`.

## Repository structure

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- config.py
|-- train.py
|-- train_original_dfmsrn.py
|-- inference.py
|-- models/
|   |-- __init__.py
|   |-- stage1.py
|   |-- stage2.py
|   |-- dfmsrn.py
|-- losses/
|   |-- __init__.py
|   |-- region_aware_loss.py
|   |-- original_loss.py
|-- datasets/
|   |-- __init__.py
|   |-- marine_snow_dataset.py
|-- utils/
    |-- __init__.py
    |-- metrics.py
    |-- image_utils.py
```

- `models/`: Stage 1, Stage 2, and the shared model wrapper.
- `losses/`: region-aware losses, original losses, and VGG19 feature extraction.
- `datasets/`: paired image and binary-mask loading; no image data are stored here.
- `utils/`: PSNR, running averages, and image-saving functions.
- `config.py`: paths, training parameters, loss weights, and inference settings.

## Installation

Install the dependencies in a Python environment:

```bash
python -m pip install -r requirements.txt
```

Dependencies are PyTorch, torchvision, NumPy, OpenCV, and tqdm. Versions are not pinned; the dependency list does not define the original experiment environment. Use a compatible PyTorch/torchvision pair and an appropriate PyTorch build for your CUDA environment.

The perceptual loss uses `torchvision.models.vgg19` with `VGG19_Weights.IMAGENET1K_V1`. The first training run downloads these weights if they are not already cached. For offline training, prepare the weights in the torchvision cache beforehand.

CUDA is selected when available; otherwise, the scripts use CPU.

## Dataset preparation

Prepare the data separately and configure the paths in `config.py`. The training scripts combine Task 1 and Task 2. The directories named `test` are used by the existing validation loop and checkpoint-selection criterion.

The expected layout is:

```text
MSRB_ROOT/
|-- training/
|   |-- MSR_Task1/
|   |-- MSR_Task1_binarymask/
|   |-- MSR_Task2/
|   |-- MSR_Task2_binarymask/
|-- test/
    |-- MSR_Task1/
    |-- MSR_Task1_binarymask/
    |-- MSR_Task2/
    |-- MSR_Task2_binarymask/

TRAIN_GT_DIR/
    image_001.png
    ...

VAL_GT_DIR/
    image_001.png
    ...
```

Each input must have a ground-truth image and, for region-aware training, a mask with exactly the same filename, including the extension. For example, `image_001.png` must be present in the corresponding input, ground-truth, and mask directories. Unmatched samples are skipped. Both tasks use the configured ground-truth directory for their split.

Images are read as RGB, converted to float32, divided by 255, and converted to CHW tensors. Masks are read as grayscale, divided by 255, and binarized using `mask > 0.5`; background is 0 and marine snow is 1. Each input, target, and mask must have matching spatial dimensions.

The loader uses full images without cropping, resizing, or augmentation. Training images must satisfy the model's spatial-resolution requirements; dimensions divisible by 16 are compatible with the multi-resolution Stage 1 pipeline. Images within each batch must have the same dimensions. Training does not apply inference-time padding.

The original-loss baseline requires input/ground-truth pairs but does not load masks. Mask creation is not part of this repository; prepare masks that match the annotations used for the intended experiment.

For MSRB data, refer to the [official dataset repository](https://github.com/ychtanaka/marine-snow). Its distributed directory names differ from the layout above. Arrange the downloaded files and separately prepared masks to match the configured layout; do not change the intended train/test split.

For experiments using WGAN-based synthetic data, prepare the paired inputs, clean targets, and binary masks separately. The provided training entrypoints use the Task 1/Task 2 layout above and do not generate synthetic data.

## Configuration

Replace the relevant `SET_*` placeholders in [config.py](config.py) before running a script. Placeholder values are not dataset or checkpoint locations.

| Setting | Purpose |
| --- | --- |
| `MSRB_ROOT` | Dataset root containing `training/` and `test/` |
| `TRAIN_GT_DIR` | Training ground-truth images |
| `VAL_GT_DIR` | Test ground-truth images used by validation |
| `CHECKPOINT_DIR` | Region-aware training checkpoints |
| `PREVIEW_DIR` | Region-aware training previews |
| `ORIGINAL_CHECKPOINT_DIR` | Original-loss baseline checkpoints |
| `ORIGINAL_PREVIEW_DIR` | Original-loss baseline previews |
| `CHECKPOINT_PATH` | Removal checkpoint for inference |
| `INPUT_DIR` | Inference input images |
| `OUTPUT_DIR` | Inference results |

Input and mask subdirectories, as well as last/best checkpoint paths, are derived from these settings. Keep the two training runs' output directories separate. No command-line arguments are implemented; configuration is read when the scripts are imported or started.

## Training

Run all commands from the repository root.

### Region-aware objective

Configure the dataset, ground-truth, checkpoint, and preview paths, then run:

```bash
python train.py
```

Stage 1 and Stage 2 are optimized jointly. The default configuration is:

| Parameter | Value |
| --- | --- |
| Random seed | 42 |
| Batch size | 4 |
| Total epochs | 80 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Adam betas | (0.9, 0.999) |
| Scheduler | StepLR |
| Scheduler step size / gamma | 5 / 0.5 |
| DataLoader workers | 4 |
| Stage 1 / Stage 2 loss weights | 0.15 / 0.85 |

The training loader shuffles samples and drops the last incomplete batch. Validation uses neither shuffling nor dropping of the last batch. Both loaders use pinned memory.

The Stage 1 objective combines whole-image MSE, Fourier amplitude MSE, snow-region Charbonnier loss, and background L1 loss. Both regional targets are the clean ground truth.

| Stage 1 component | Weight |
| --- | --- |
| Image MSE | 1.0 |
| Fourier amplitude MSE | 1.0 |
| Snow-region Charbonnier | 3.0 |
| Background L1 | 2.0 |

The Stage 2 objective combines whole-image MSE, VGG19 perceptual loss, snow-region Charbonnier loss against clean ground truth, and background identity L1 loss against the original input.

| Stage 2 component | Weight |
| --- | --- |
| Image MSE | 1.0 |
| VGG19 perceptual | 0.1 |
| Snow-region Charbonnier | 2.0 |
| Background identity L1 | 5.0 |

Stage 2 dilates the binary snow mask in memory using a 3x3 max-pooling operation before computing the regional losses. This also applies during validation. Source mask files are unchanged. The Charbonnier epsilon is `1e-3`.

### Original-loss baseline

Configure the dataset, ground-truth, and baseline output paths, then run:

```bash
python train_original_dfmsrn.py
```

This baseline uses the same Stage 1 and residual Stage 2 architecture. Its Stage 1 loss is image MSE plus Fourier amplitude MSE. Its Stage 2 loss is image MSE plus 0.1 times VGG19 perceptual loss. The total objective retains the 0.15/0.85 stage weights and does not use binary masks.

### Checkpoints and training outputs

| Run | Last checkpoint | Best checkpoint |
| --- | --- | --- |
| Region-aware | `CHECKPOINT_DIR/dfmsrn_last.pth` | `CHECKPOINT_DIR/dfmsrn_best.pth` |
| Original-loss | `ORIGINAL_CHECKPOINT_DIR/dfmsrn_original_loss_last.pth` | `ORIGINAL_CHECKPOINT_DIR/dfmsrn_original_loss_best.pth` |

The best checkpoint minimizes validation total loss. Training checkpoints contain model, Stage 1, Stage 2, optimizer, and scheduler state dictionaries, the epoch, and loss/metric metadata.

Region-aware training defaults to `RESUME_TRAINING = True`. If its last checkpoint exists, training restores model, optimizer, and scheduler states and resumes at the next epoch, up to the configured total of 80 epochs. Otherwise, it starts from scratch. Set `RESUME_TRAINING = False` for a fresh region-aware run. The baseline always starts from scratch. Random-number-generator states are not stored in checkpoints, so resumed runs are not guaranteed to reproduce an uninterrupted run exactly.

Each epoch saves a fixed-sample preview under the corresponding preview directory in an `epoch_NNN/` subdirectory. Outputs include input, intermediate prediction, final prediction, ground truth, a comparison grid, and PSNR statistics. Region-aware previews also include the mask.

## Inference

Prepare a removal-network checkpoint separately. Configure `CHECKPOINT_PATH`, `INPUT_DIR`, and `OUTPUT_DIR`, then run:

```bash
python inference.py
```

The loader accepts either a training checkpoint containing `model_state_dict` or a raw full-model state dictionary. Weights are loaded with `strict=True`. The checkpoint must match the complete architecture in `models/`; a WGAN generator checkpoint is not compatible.

Inference reads supported images directly from `INPUT_DIR` without recursively searching subdirectories. Supported extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.

Inputs are converted to RGB float32 in [0, 1]. Reflection padding is added on the right and bottom to obtain dimensions divisible by 16. After inference, outputs are cropped back to the original size. Predictions are clamped to [0, 1], rounded, and converted to uint8 only when saving. Very small images may not satisfy reflection-padding or normalization constraints.

```text
OUTPUT_DIR/
|-- x_fin/     # Final restored PNG images
|-- x_mid/     # Intermediate PNG images when SAVE_XMID = True
```

`SAVE_XMID` defaults to `True`. Outputs use the input filename stem with a `.png` extension; input images must have distinct stems to avoid overwriting results.

## References and acknowledgments

This work builds on Fourier-based marine snow removal and half instance normalization. Relevant prior work and resources are:

- Yakun Ju et al., **Towards marine snow removal with fusing Fourier information**, Information Fusion, 117, 102810, 2025. [Paper](https://www.sciencedirect.com/science/article/pii/S1566253524005888) · [DOI](https://doi.org/10.1016/j.inffus.2024.102810).
- Liangyu Chen et al., **HINet: Half Instance Normalization Network for Image Restoration**, CVPR Workshops, 2021. [Paper](https://arxiv.org/abs/2105.06086) · [Repository](https://github.com/megvii-model/HINet).
- **Marine Snow Removal Benchmarking Dataset (MSRB)**. [Dataset and image credits](https://github.com/ychtanaka/marine-snow).
- **torchvision VGG19**, used for perceptual feature extraction. [Repository](https://github.com/pytorch/vision).

## License and data terms

A project license has not been specified. Third-party software, datasets, and pretrained weights remain subject to their respective terms. HINet's repository includes MIT and BasicSR Apache-2.0 notices; torchvision is distributed under BSD-3-Clause. The MSRB repository identifies its source images as CC BY-NC-SA 2.0 and provides image credits. Consult the upstream sources before redistributing their code, images, derived data, or weights.
