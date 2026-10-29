import random
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from config import (
    MSRB_ROOT,
    TRAIN_TASK1_INPUT_DIR,
    TRAIN_TASK1_MASK_DIR,
    TRAIN_TASK2_INPUT_DIR,
    TRAIN_TASK2_MASK_DIR,
    TRAIN_GT_DIR,
    VAL_TASK1_INPUT_DIR,
    VAL_TASK1_MASK_DIR,
    VAL_TASK2_INPUT_DIR,
    VAL_TASK2_MASK_DIR,
    VAL_GT_DIR,
    CHECKPOINT_DIR,
    PREVIEW_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    BETA1,
    BETA2,
    LR_STEP_SIZE,
    LR_GAMMA,
    NUM_WORKERS,
    SEED,
    DEVICE,
    RESUME_TRAINING,
    RESUME_CHECKPOINT,
    BEST_CHECKPOINT,
    PHI,
    STAGE1_WEIGHT,
    STAGE2_WEIGHT,
    IMAGE_LOSS_WEIGHT,
    FOURIER_LOSS_WEIGHT,
    SNOW_LOSS_WEIGHT,
    BACKGROUND_LOSS_WEIGHT,
    CHARBONNIER_EPS,
    PERCEPTUAL_WEIGHT,
    STAGE2_SNOW_LOSS_WEIGHT,
    STAGE2_BACKGROUND_LOSS_WEIGHT,
    MASK_DILATION_KERNEL,
    IMAGE_EXTENSIONS,
)
from models import DFMSRN
from datasets.marine_snow_dataset import PairedMarineSnowDataset
from losses.region_aware_loss import Stage1CustomLoss, Stage2PaperLoss
from utils.metrics import calculate_psnr, AverageMeter
from utils.image_utils import tensor_to_rgb_uint8, save_tensor_image, save_mask

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

@torch.no_grad()

def save_preview(model, val_dataset, epoch):

    model.eval()

    (input_image, gt_image, snow_mask, filename) = val_dataset[0]

    input_image = input_image.unsqueeze(0).to(DEVICE)

    gt_image = gt_image.unsqueeze(0).to(DEVICE)

    snow_mask = snow_mask.unsqueeze(0).to(DEVICE)

    outputs = model(input_image)

    x_mid = outputs['x_mid']

    x_fin = outputs['x_fin']

    epoch_dir = PREVIEW_DIR / f'epoch_{epoch:03d}'

    epoch_dir.mkdir(parents=True, exist_ok=True)

    save_tensor_image(input_image, epoch_dir / '01_input.png')

    save_tensor_image(x_mid, epoch_dir / '02_x_mid.png')

    save_tensor_image(x_fin, epoch_dir / '03_x_fin.png')

    save_tensor_image(gt_image, epoch_dir / '04_gt.png')

    save_mask(snow_mask, epoch_dir / '05_mask.png')

    input_np = tensor_to_rgb_uint8(input_image)

    mid_np = tensor_to_rgb_uint8(x_mid)

    fin_np = tensor_to_rgb_uint8(x_fin)

    gt_np = tensor_to_rgb_uint8(gt_image)

    grid = np.concatenate((input_np, mid_np, fin_np, gt_np), axis=1)

    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(epoch_dir / 'preview_grid.png'), grid_bgr)

    input_psnr = calculate_psnr(input_image, gt_image)

    mid_psnr = calculate_psnr(x_mid, gt_image)

    fin_psnr = calculate_psnr(x_fin, gt_image)

    with open(
        epoch_dir
        / "stats.txt",
        "w",
    ) as f:

        f.write(f'Filename: {filename}\n\n')

        f.write(f'Input PSNR : {input_psnr:.3f} dB\n')

        f.write(f'X_mid PSNR : {mid_psnr:.3f} dB\n')

        f.write(f'X_fin PSNR : {fin_psnr:.3f} dB\n')

    return {'input_psnr': input_psnr, 'mid_psnr': mid_psnr, 'fin_psnr': fin_psnr}

def train_one_epoch(
    model,
    loader,
    stage1_criterion,
    stage2_criterion,
    optimizer,
    epoch,
):

    model.train()

    # Keep the perceptual feature extractor in evaluation mode.

    stage2_criterion.vgg.eval()

    meters = {
        "total":
            AverageMeter(),
        "l1":
            AverageMeter(),
        "l1_image":
            AverageMeter(),
        "l1_fourier":
            AverageMeter(),
        "l1_snow":
            AverageMeter(),
        "l1_background":
            AverageMeter(),
        "l2":
            AverageMeter(),
        "l2_image":
            AverageMeter(),
        "l2_perceptual":
            AverageMeter(),
        "l2_snow":
            AverageMeter(),
        "l2_background":
            AverageMeter(),
    }

    progress = tqdm(loader, desc=f'Train {epoch:03d}')

    for (
        input_image,
        gt_image,
        snow_mask,
        _,
    ) in progress:

        input_image = input_image.to(DEVICE, non_blocking=True)

        gt_image = gt_image.to(DEVICE, non_blocking=True)

        snow_mask = snow_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_image)

        x_mid = outputs['x_mid']

        x_fin = outputs['x_fin']

        l1 = stage1_criterion(x_mid=x_mid, gt=gt_image, snow_mask=snow_mask)

        # Stage 2 Loss

        l2 = stage2_criterion(
            x_fin=x_fin,
            gt=gt_image,
            input_image=input_image,
            snow_mask=snow_mask,
        )

        total_loss = STAGE1_WEIGHT * l1['total'] + STAGE2_WEIGHT * l2['total']

        total_loss.backward()

        optimizer.step()

        batch_size = input_image.shape[0]

        meters['total'].update(total_loss.item(), batch_size)

        meters['l1'].update(l1['total'].item(), batch_size)

        meters['l1_image'].update(l1['image'].item(), batch_size)

        meters['l1_fourier'].update(l1['fourier'].item(), batch_size)

        meters['l1_snow'].update(l1['snow'].item(), batch_size)

        meters['l1_background'].update(l1['background'].item(), batch_size)

        meters['l2'].update(l2['total'].item(), batch_size)

        meters['l2_image'].update(l2['image'].item(), batch_size)

        meters['l2_perceptual'].update(l2['perceptual'].item(), batch_size)

        meters['l2_snow'].update(l2['snow'].item(), batch_size)

        meters['l2_background'].update(l2['background'].item(), batch_size)

        progress.set_postfix(
            total=
                f"{meters['total'].avg:.4f}",
            l1=
                f"{meters['l1'].avg:.4f}",
            l2=
                f"{meters['l2'].avg:.4f}",
        )

    return {key: meter.avg for (key, meter) in meters.items()}

@torch.no_grad()

def validate(model, loader, stage1_criterion, stage2_criterion, epoch):

    model.eval()

    stage2_criterion.vgg.eval()

    meters = {
        "total":
            AverageMeter(),
        "l1":
            AverageMeter(),
        "l1_image":
            AverageMeter(),
        "l1_fourier":
            AverageMeter(),
        "l1_snow":
            AverageMeter(),
        "l1_background":
            AverageMeter(),
        "l2":
            AverageMeter(),
        "l2_image":
            AverageMeter(),
        "l2_perceptual":
            AverageMeter(),
        "l2_snow":
            AverageMeter(),
        "l2_background":
            AverageMeter(),
        "mid_psnr":
            AverageMeter(),
        "fin_psnr":
            AverageMeter(),
    }

    progress = tqdm(loader, desc=f'Val   {epoch:03d}')

    for (
        input_image,
        gt_image,
        snow_mask,
        _,
    ) in progress:

        input_image = input_image.to(DEVICE, non_blocking=True)

        gt_image = gt_image.to(DEVICE, non_blocking=True)

        snow_mask = snow_mask.to(DEVICE, non_blocking=True)

        outputs = model(input_image)

        x_mid = outputs['x_mid']

        x_fin = outputs['x_fin']

        l1 = stage1_criterion(x_mid=x_mid, gt=gt_image, snow_mask=snow_mask)

        l2 = stage2_criterion(
            x_fin=x_fin,
            gt=gt_image,
            input_image=input_image,
            snow_mask=snow_mask,
        )

        total_loss = STAGE1_WEIGHT * l1['total'] + STAGE2_WEIGHT * l2['total']

        batch_size = input_image.shape[0]

        meters['total'].update(total_loss.item(), batch_size)

        meters['l1'].update(l1['total'].item(), batch_size)

        meters['l1_image'].update(l1['image'].item(), batch_size)

        meters['l1_fourier'].update(l1['fourier'].item(), batch_size)

        meters['l1_snow'].update(l1['snow'].item(), batch_size)

        meters['l1_background'].update(l1['background'].item(), batch_size)

        meters['l2'].update(l2['total'].item(), batch_size)

        meters['l2_image'].update(l2['image'].item(), batch_size)

        meters['l2_perceptual'].update(l2['perceptual'].item(), batch_size)

        meters['l2_snow'].update(l2['snow'].item(), batch_size)

        meters['l2_background'].update(l2['background'].item(), batch_size)

        for b in range(
            batch_size
        ):

            mid_psnr = calculate_psnr(x_mid[b:b + 1], gt_image[b:b + 1])

            fin_psnr = calculate_psnr(x_fin[b:b + 1], gt_image[b:b + 1])

            meters['mid_psnr'].update(mid_psnr)

            meters['fin_psnr'].update(fin_psnr)

        progress.set_postfix(
            total=
                f"{meters['total'].avg:.4f}",
            mid=
                f"{meters['mid_psnr'].avg:.2f}",
            fin=
                f"{meters['fin_psnr'].avg:.2f}",
        )

    return {key: meter.avg for (key, meter) in meters.items()}

def print_epoch_stats(epoch, train_stats, val_stats, lr, epoch_time):

    print()

    print("=" * 70)

    print(f'[Epoch {epoch:03d}/{EPOCHS:03d}]')

    print("-" * 70)

    print(f'LR               : {lr:.8f}')

    print(f'Epoch Time       : {epoch_time:.2f} sec')

    print()

    print(f"Train Total      : {train_stats['total']:.6f}")

    print(f"Train L1         : {train_stats['l1']:.6f}")

    print(f"Train L2         : {train_stats['l2']:.6f}")

    print()

    print(f"Train L1 Image   : {train_stats['l1_image']:.6f}")

    print(f"Train L1 Fourier : {train_stats['l1_fourier']:.6f}")

    print(f"Train L1 Snow    : {train_stats['l1_snow']:.6f}")

    print(f"Train L1 Bg      : {train_stats['l1_background']:.6f}")

    print()

    print(f"Train L2 Image   : {train_stats['l2_image']:.6f}")

    print(f"Train L2 VGG     : {train_stats['l2_perceptual']:.6f}")

    print(f"Train L2 Snow    : {train_stats['l2_snow']:.6f}")

    print(f"Train L2 Bg      : {train_stats['l2_background']:.6f}")

    print()

    print("-" * 70)

    print(f"Val Total        : {val_stats['total']:.6f}")

    print(f"Val L1           : {val_stats['l1']:.6f}")

    print(f"Val L2           : {val_stats['l2']:.6f}")

    print()

    print(f"Val L1 Image     : {val_stats['l1_image']:.6f}")

    print(f"Val L1 Fourier   : {val_stats['l1_fourier']:.6f}")

    print(f"Val L1 Snow      : {val_stats['l1_snow']:.6f}")

    print(f"Val L1 Bg        : {val_stats['l1_background']:.6f}")

    print()

    print(f"Val L2 Image     : {val_stats['l2_image']:.6f}")

    print(f"Val L2 VGG       : {val_stats['l2_perceptual']:.6f}")

    print(f"Val L2 Snow      : {val_stats['l2_snow']:.6f}")

    print(f"Val L2 Bg        : {val_stats['l2_background']:.6f}")

    print()

    print(f"Val X_mid PSNR   : {val_stats['mid_psnr']:.3f} dB")

    print(f"Val X_fin PSNR   : {val_stats['fin_psnr']:.3f} dB")

    print("=" * 70)

    print()

def main():

    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    train_task1_dataset = (
        PairedMarineSnowDataset(
            input_dir=
                TRAIN_TASK1_INPUT_DIR,
            gt_dir=
                TRAIN_GT_DIR,
            mask_dir=
                TRAIN_TASK1_MASK_DIR,
            training=True,
            dataset_name=
                "TRAIN TASK 1",
        )
    )

    train_task2_dataset = (
        PairedMarineSnowDataset(
            input_dir=
                TRAIN_TASK2_INPUT_DIR,
            gt_dir=
                TRAIN_GT_DIR,
            mask_dir=
                TRAIN_TASK2_MASK_DIR,
            training=True,
            dataset_name=
                "TRAIN TASK 2",
        )
    )

    train_dataset = ConcatDataset([train_task1_dataset, train_task2_dataset])

    val_task1_dataset = (
        PairedMarineSnowDataset(
            input_dir=
                VAL_TASK1_INPUT_DIR,
            gt_dir=
                VAL_GT_DIR,
            mask_dir=
                VAL_TASK1_MASK_DIR,
            training=False,
            dataset_name=
                "VALIDATION TASK 1",
        )
    )

    val_task2_dataset = (
        PairedMarineSnowDataset(
            input_dir=
                VAL_TASK2_INPUT_DIR,
            gt_dir=
                VAL_GT_DIR,
            mask_dir=
                VAL_TASK2_MASK_DIR,
            training=False,
            dataset_name=
                "VALIDATION TASK 2",
        )
    )

    val_dataset = ConcatDataset([val_task1_dataset, val_task2_dataset])

    print()

    print("=" * 70)

    print('COMBINED MSRB DATASET')

    print("=" * 70)

    print(f'Train Task1       : {len(train_task1_dataset)}')

    print(f'Train Task2       : {len(train_task2_dataset)}')

    print(f'Train Total       : {len(train_dataset)}')

    print()

    print(f'Val Task1         : {len(val_task1_dataset)}')

    print(f'Val Task2         : {len(val_task2_dataset)}')

    print(f'Val Total         : {len(val_dataset)}')

    print("=" * 70)

    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=True,
        num_workers=
            NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
        num_workers=
            NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    model = DFMSRN().to(DEVICE)

    stage1_criterion = Stage1CustomLoss().to(DEVICE)

    stage2_criterion = Stage2PaperLoss().to(DEVICE)

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=
                LEARNING_RATE,
            betas=(
                BETA1,
                BETA2,
            ),
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=
                LR_STEP_SIZE,
            gamma=
                LR_GAMMA,
        )
    )

    total_parameters = sum((p.numel() for p in model.parameters()))

    trainable_parameters = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    print()

    print("=" * 70)

    print('DF-MSRN MSRB FULL JOINT TRAINING')

    print("=" * 70)

    print(f'Device            : {DEVICE}')

    if torch.cuda.is_available():

        print(f'GPU               : {torch.cuda.get_device_name(0)}')

    print(f'Train Task1       : {len(train_task1_dataset)}')

    print(f'Train Task2       : {len(train_task2_dataset)}')

    print(f'Train Total       : {len(train_dataset)}')

    print(f'Val Task1         : {len(val_task1_dataset)}')

    print(f'Val Task2         : {len(val_task2_dataset)}')

    print(f'Val Total         : {len(val_dataset)}')

    print(f'Epochs            : {EPOCHS}')

    print(f'Batch             : {BATCH_SIZE}')

    print(f'Learning Rate     : {LEARNING_RATE}')

    print(f'L1 Weight         : {STAGE1_WEIGHT}')

    print(f'L2 Weight         : {STAGE2_WEIGHT}')

    print(f'Snow Loss Weight  : {SNOW_LOSS_WEIGHT}')

    print(f'Background Weight : {BACKGROUND_LOSS_WEIGHT}')

    print(f'VGG Weight        : {PERCEPTUAL_WEIGHT}')

    print(f'Stage2 Snow       : {STAGE2_SNOW_LOSS_WEIGHT}')

    print(f'Stage2 Background : {STAGE2_BACKGROUND_LOSS_WEIGHT}')

    print(f'Mask Dilation     : {MASK_DILATION_KERNEL}x{MASK_DILATION_KERNEL}')

    print(f'Parameters        : {total_parameters:,}')

    print(f'Trainable         : {trainable_parameters:,}')

    # Restore model, optimizer and scheduler for resumed training.

    start_epoch = 1

    best_val_total = float('inf')

    resumed_from_epoch = None

    if (
        RESUME_TRAINING
        and RESUME_CHECKPOINT.exists()
    ):

        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        resumed_from_epoch = int(checkpoint['epoch'])

        start_epoch = resumed_from_epoch + 1

        # Checkpoints are saved before scheduler.step().
        # Advance the restored scheduler to the completed epoch.

        while (
            scheduler.last_epoch
            < resumed_from_epoch
        ):

            scheduler.step()

        if BEST_CHECKPOINT.exists():

            best_checkpoint = torch.load(BEST_CHECKPOINT, map_location='cpu')

            best_val_total = float(best_checkpoint.get('val_total_loss', float('inf')))

        else:

            best_val_total = float(checkpoint.get('val_total_loss', float('inf')))

    print()

    if resumed_from_epoch is None:

        print('Checkpoint loading: NONE')

        print('Training start    : FROM SCRATCH')

    else:

        print(f'Checkpoint loading: {RESUME_CHECKPOINT}')

        print(f'Resume epoch      : {resumed_from_epoch}')

        print(f'Training start    : EPOCH {start_epoch}')

        print(f"Resume LR         : {optimizer.param_groups[0]['lr']:.8f}")

        print(f'Previous BEST loss: {best_val_total:.6f}')

    print('Stage 1 + Stage 2: JOINT')

    print('Best criterion   : minimum validation TOTAL loss')

    print("=" * 70)

    print()

    if start_epoch > EPOCHS:

        print(f'Checkpoint epoch ({start_epoch - 1}) is greater than or equal to EPOCHS ({EPOCHS}).')

        print('Increase EPOCHS to continue training.')

        return

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):

        if DEVICE == "cuda":

            torch.cuda.synchronize()

        epoch_start = time.time()

        train_stats = (
            train_one_epoch(
                model=model,
                loader=
                    train_loader,
                stage1_criterion=
                    stage1_criterion,
                stage2_criterion=
                    stage2_criterion,
                optimizer=
                    optimizer,
                epoch=
                    epoch,
            )
        )

        val_stats = (
            validate(
                model=model,
                loader=
                    val_loader,
                stage1_criterion=
                    stage1_criterion,
                stage2_criterion=
                    stage2_criterion,
                epoch=
                    epoch,
            )
        )

        if DEVICE == "cuda":

            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]['lr']

        print_epoch_stats(
            epoch=
                epoch,
            train_stats=
                train_stats,
            val_stats=
                val_stats,
            lr=
                current_lr,
            epoch_time=
                epoch_time,
        )

        preview_stats = save_preview(model=model, val_dataset=val_dataset, epoch=epoch)

        print('Fixed Preview')

        print(f"Input : {preview_stats['input_psnr']:.3f} dB")

        print(f"X_mid : {preview_stats['mid_psnr']:.3f} dB")

        print(f"X_fin : {preview_stats['fin_psnr']:.3f} dB")

        print()

        checkpoint = {
            "epoch":
                epoch,
            "model_state_dict":
                model.state_dict(),
            "stage1_state_dict":
                model.stage1.state_dict(),
            "stage2_state_dict":
                model.stage2.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "train_total_loss":
                train_stats[
                    "total"
                ],
            "val_total_loss":
                val_stats[
                    "total"
                ],
            "val_l1_loss":
                val_stats[
                    "l1"
                ],
            "val_l2_loss":
                val_stats[
                    "l2"
                ],
            "val_mid_psnr":
                val_stats[
                    "mid_psnr"
                ],
            "val_fin_psnr":
                val_stats[
                    "fin_psnr"
                ],
            "best_val_total":
                min(
                    best_val_total,
                    val_stats[
                        "total"
                    ],
                ),
            "phi":
                PHI,
            "stage1_weight":
                STAGE1_WEIGHT,
            "stage2_weight":
                STAGE2_WEIGHT,
            "image_loss_weight":
                IMAGE_LOSS_WEIGHT,
            "fourier_loss_weight":
                FOURIER_LOSS_WEIGHT,
            "snow_loss_weight":
                SNOW_LOSS_WEIGHT,
            "background_loss_weight":
                BACKGROUND_LOSS_WEIGHT,
            "charbonnier_eps":
                CHARBONNIER_EPS,
            "perceptual_weight":
                PERCEPTUAL_WEIGHT,
            "stage2_snow_loss_weight":
                STAGE2_SNOW_LOSS_WEIGHT,
            "stage2_background_loss_weight":
                STAGE2_BACKGROUND_LOSS_WEIGHT,
            "mask_dilation_kernel":
                MASK_DILATION_KERNEL,
            "stage1_loss_type":
                (
                    "image_mse"
                    " + fourier_amplitude_mse"
                    f" + {SNOW_LOSS_WEIGHT:g}*snow_charbonnier"
                    " + background_l1"
                ),
            "stage2_loss_type":
                (
                    "image_mse"
                    " + 0.1*vgg19_perceptual"
                    f" + {STAGE2_SNOW_LOSS_WEIGHT:g}*dilated_snow_charbonnier"
                    f" + {STAGE2_BACKGROUND_LOSS_WEIGHT:g}*background_identity_l1"
                ),
            "total_loss_type":
                (
                    "0.15*L1"
                    " + 0.85*L2"
                ),
            "dataset":
                (
                    "MSRB Task1 + Task2"
                ),
        }

        last_path = CHECKPOINT_DIR / 'dfmsrn_last.pth'

        torch.save(checkpoint, last_path)

        if (
            val_stats[
                "total"
            ]
            < best_val_total
        ):

            best_val_total = val_stats['total']

            best_path = CHECKPOINT_DIR / 'dfmsrn_best.pth'

            torch.save(checkpoint, best_path)

            print()

            print('★ New BEST DF-MSRN')

            print(f'  Epoch          : {epoch}')

            print(f'  Val Total Loss : {best_val_total:.6f}')

            print(f"  X_mid PSNR     : {val_stats['mid_psnr']:.3f} dB")

            print(f"  X_fin PSNR     : {val_stats['fin_psnr']:.3f} dB")

            print()

        scheduler.step()

        print(f"Preview path : {PREVIEW_DIR / f'epoch_{epoch:03d}'}")

        print(f"Last path    : {CHECKPOINT_DIR / 'dfmsrn_last.pth'}")

        print(f"Best path    : {CHECKPOINT_DIR / 'dfmsrn_best.pth'}")

        print(f"Next LR      : {optimizer.param_groups[0]['lr']:.8f}")

        print()

    print()

    print("=" * 70)

    print('MSRB Training Finished')

    print("=" * 70)

    print(f'Final Epoch                : {EPOCHS}')

    print(f'Best Validation Total Loss : {best_val_total:.6f}')

    print(f"Last Checkpoint            : {CHECKPOINT_DIR / 'dfmsrn_last.pth'}")

    print(f"Best Checkpoint            : {CHECKPOINT_DIR / 'dfmsrn_best.pth'}")

    print("=" * 70)

    print()

if __name__ == "__main__":
    main()
