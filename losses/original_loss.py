import torch
import torch.nn as nn
from models.stage1 import fft_decompose
from config import PERCEPTUAL_WEIGHT
from .region_aware_loss import VGG19FeatureExtractor

class Stage1OriginalLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, x_mid, gt):

        image_loss = self.mse(x_mid, gt)

        (gt_amp, _) = fft_decompose(gt)

        (restored_amp, _) = fft_decompose(x_mid)

        fourier_loss = self.mse(restored_amp, gt_amp)

        total = image_loss + fourier_loss

        return {'total': total, 'image': image_loss, 'fourier': fourier_loss}

class Stage2OriginalLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

        self.vgg = VGG19FeatureExtractor()

    def forward(self, x_fin, gt):

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

        total = image_loss + PERCEPTUAL_WEIGHT * perceptual_loss

        return {'total': total, 'image': image_loss, 'perceptual': perceptual_loss}
