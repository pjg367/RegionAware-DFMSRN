import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from models.stage1 import fft_decompose
from config import (
    IMAGE_LOSS_WEIGHT,
    FOURIER_LOSS_WEIGHT,
    SNOW_LOSS_WEIGHT,
    BACKGROUND_LOSS_WEIGHT,
    CHARBONNIER_EPS,
    PERCEPTUAL_WEIGHT,
    STAGE2_SNOW_LOSS_WEIGHT,
    STAGE2_BACKGROUND_LOSS_WEIGHT,
    MASK_DILATION_KERNEL,
)

class Stage1CustomLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

    @staticmethod

    def masked_charbonnier(prediction, target, mask):

        difference = prediction - target

        charbonnier = torch.sqrt(difference ** 2 + CHARBONNIER_EPS ** 2)

        masked = charbonnier * mask

        denominator = mask.sum() * prediction.shape[1]

        return masked.sum() / (denominator + 1e-08)

    @staticmethod

    def masked_l1(prediction, target, mask):

        difference = torch.abs(prediction - target)

        masked = difference * mask

        denominator = mask.sum() * prediction.shape[1]

        return masked.sum() / (denominator + 1e-08)

    def forward(self, x_mid, gt, snow_mask):

        image_loss = self.mse(x_mid, gt)

        (gt_amp, _) = fft_decompose(gt)

        (restored_amp, _) = fft_decompose(x_mid)

        fourier_loss = self.mse(restored_amp, gt_amp)

        snow_loss = self.masked_charbonnier(x_mid, gt, snow_mask)

        background_mask = 1.0 - snow_mask

        background_loss = self.masked_l1(x_mid, gt, background_mask)

        total = (
            IMAGE_LOSS_WEIGHT
            * image_loss
            +
            FOURIER_LOSS_WEIGHT
            * fourier_loss
            +
            SNOW_LOSS_WEIGHT
            * snow_loss
            +
            BACKGROUND_LOSS_WEIGHT
            * background_loss
        )

        return {
            "total":
                total,
            "image":
                image_loss,
            "fourier":
                fourier_loss,
            "snow":
                snow_loss,
            "background":
                background_loss,
        }

class VGG19FeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        weights = VGG19_Weights.IMAGENET1K_V1

        network = vgg19(weights=weights).features

        self.blocks = (
            nn.ModuleList([
                network[:4],
                network[4:9],
                network[9:18],
                network[18:27],
            ])
        )

        for parameter in (
            self.parameters()
        ):

            parameter.requires_grad = False

        self.eval()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer('mean', mean)

        self.register_buffer('std', std)

    def forward(self, x):

        x = torch.clamp(x, 0.0, 1.0)

        x = (x - self.mean) / self.std

        features = []

        for block in self.blocks:

            x = block(x)

            features.append(x)

        return features

class Stage2PaperLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

        self.vgg = VGG19FeatureExtractor()

    @staticmethod
    def dilate_mask(mask, kernel_size):

        if kernel_size <= 1:
            return mask

        if kernel_size % 2 == 0:
            raise ValueError('MASK_DILATION_KERNEL must be odd.')

        padding = kernel_size // 2

        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)

        return torch.clamp(dilated, 0.0, 1.0)

    @staticmethod
    def masked_charbonnier(prediction, target, mask):

        difference = prediction - target

        charbonnier = torch.sqrt(difference ** 2 + CHARBONNIER_EPS ** 2)

        masked = charbonnier * mask

        denominator = mask.sum() * prediction.shape[1]

        return masked.sum() / (denominator + 1e-08)

    @staticmethod
    def masked_l1(prediction, target, mask):

        difference = torch.abs(prediction - target)

        masked = difference * mask

        denominator = mask.sum() * prediction.shape[1]

        return masked.sum() / (denominator + 1e-08)

    def forward(self, x_fin, gt, input_image, snow_mask):

        image_loss = self.mse(x_fin, gt)

        with torch.no_grad():

            gt_features = self.vgg(gt)

        pred_features = self.vgg(x_fin)

        perceptual_loss = torch.zeros((), device=x_fin.device, dtype=x_fin.dtype)

        for (
            pred_feature,
            gt_feature,
        ) in zip(
            pred_features,
            gt_features,
        ):

            perceptual_loss = perceptual_loss + self.mse(pred_feature, gt_feature)

        perceptual_loss = perceptual_loss / len(pred_features)

        dilated_snow_mask = self.dilate_mask(snow_mask, MASK_DILATION_KERNEL)

        background_mask = 1.0 - dilated_snow_mask

        # Compare the dilated snow region with clean GT.

        snow_loss = self.masked_charbonnier(x_fin, gt, dilated_snow_mask)

        # Compare the background with the original input.

        background_loss = self.masked_l1(x_fin, input_image, background_mask)

        total = (
            image_loss
            +
            PERCEPTUAL_WEIGHT
            * perceptual_loss
            +
            STAGE2_SNOW_LOSS_WEIGHT
            * snow_loss
            +
            STAGE2_BACKGROUND_LOSS_WEIGHT
            * background_loss
        )

        return {
            "total":
                total,
            "image":
                image_loss,
            "perceptual":
                perceptual_loss,
            "snow":
                snow_loss,
            "background":
                background_loss,
        }
