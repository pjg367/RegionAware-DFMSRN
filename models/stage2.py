import torch
import torch.nn as nn
import torch.nn.functional as F

# Config

FFT_NORM = "ortho"

FEATURE_CHANNELS = 32

HFE_KERNEL_SIZE = 5
HFE_EPS = 1e-6

def fft_decompose(image):

    fourier = torch.fft.fft2(image, dim=(-2, -1), norm=FFT_NORM)

    amplitude = torch.abs(fourier)

    phase = torch.angle(fourier)

    return amplitude, phase

def ifft_reconstruct(amplitude, phase):

    real = amplitude * torch.cos(phase)

    imag = amplitude * torch.sin(phase)

    complex_fourier = torch.complex(real, imag)

    image = torch.fft.ifft2(complex_fourier, dim=(-2, -1), norm=FFT_NORM)

    return image.real

# Half Instance Normalization

class HINBlock(nn.Module):
    """
    Half Instance Normalization.

    Feature channels are divided into two halves.
    Only one half is instance-normalized.
    """

    def __init__(self, channels):
        super().__init__()

        if channels % 2 != 0:
            raise ValueError('HIN channels must be even.')

        half_channels = channels // 2

        self.instance_norm = nn.InstanceNorm2d(half_channels, affine=True)

    def forward(self, x):

        (x1, x2) = torch.chunk(x, chunks=2, dim=1)

        x1 = self.instance_norm(x1)

        return torch.cat((x1, x2), dim=1)

class SpatialResidualBlock(nn.Module):
    """
    Paper:
        HIN path
        parallel 3x3 Conv + LeakyReLU path
        residual manner
    """

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.hin = HINBlock(channels)

        self.activation1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.activation2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.hin(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.activation2(x)

        return x + residual

class SpatialDomainModule(nn.Module):

    def __init__(self, channels=FEATURE_CHANNELS):
        super().__init__()

        # H/2 feature resolution
        # This corresponds to the feature resolution that
        # Theta's stride=2 convolution must match.

        self.entry = nn.Sequential(
            nn.Conv2d(
                3,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        self.block1 = SpatialResidualBlock(channels)

        self.block2 = SpatialResidualBlock(channels)

    def forward(self, x):

        x = self.entry(x)

        x = self.block1(x)
        x = self.block2(x)

        return x

class FourierProcessingBlock(nn.Module):
    """
    FFT
      ↓
    Amplitude / Phase
      ↓
    two convolution layers for each
      ↓
    IFFT
    """

    def __init__(self, channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.amp_conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

        self.amp_conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

        self.phase_conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

        self.phase_conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        (amplitude, phase) = fft_decompose(x)

        amplitude = self.amp_conv1(amplitude)

        amplitude = self.activation(amplitude)

        amplitude = self.amp_conv2(amplitude)

        amplitude = self.activation(amplitude)

        # Fourier amplitude must remain non-negative.
        amplitude = F.softplus(amplitude)

        phase = self.phase_conv1(phase)

        phase = self.activation(phase)

        phase = self.phase_conv2(phase)

        # Do not apply softplus to phase.
        # Phase can be positive or negative.

        return ifft_reconstruct(amplitude, phase)

class FourierDomainModule(nn.Module):

    def __init__(self, channels=FEATURE_CHANNELS):
        super().__init__()

        # Match SD feature resolution: H/2 x W/2

        self.entry = nn.Sequential(
            nn.Conv2d(
                3,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        self.fourier_block1 = FourierProcessingBlock(channels=channels, kernel_size=3)

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        self.fourier_block2 = FourierProcessingBlock(channels=channels, kernel_size=1)

    def forward(self, x):

        x = self.entry(x)

        x = self.fourier_block1(x)

        x = self.spatial_conv(x)

        x = self.fourier_block2(x)

        return x

def gaussian_kernel_2d(kernel_size, sigma, device, dtype):

    coords = torch.arange(kernel_size, device=device, dtype=dtype)

    coords = coords - (kernel_size - 1) / 2.0

    gaussian_1d = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))

    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)

    gaussian_2d = gaussian_2d / gaussian_2d.sum()

    return gaussian_2d

def gaussian_blur_5x5(x):

    channels = x.shape[1]

    kernel = gaussian_kernel_2d(
        kernel_size=HFE_KERNEL_SIZE,
        sigma=1.0,
        device=x.device,
        dtype=x.dtype,
    )

    kernel = kernel.view(1, 1, HFE_KERNEL_SIZE, HFE_KERNEL_SIZE)

    kernel = kernel.repeat(channels, 1, 1, 1)

    x_padded = F.pad(
        x,
        (
            HFE_KERNEL_SIZE // 2,
            HFE_KERNEL_SIZE // 2,
            HFE_KERNEL_SIZE // 2,
            HFE_KERNEL_SIZE // 2,
        ),
        mode="reflect",
    )

    blurred = F.conv2d(x_padded, kernel, groups=channels)

    return blurred

def high_frequency_extraction(x):
    """
    Paper Eq. (3)

        HFE(X) =
            blur(X) / |blur(X) - X|

    Numerical epsilon is added only to avoid division by zero.
    """

    blurred = gaussian_blur_5x5(x)

    denominator = torch.abs(blurred - x)

    hfe = blurred / (denominator + HFE_EPS)

    return hfe

# HFE Fusion Module

class HFEBasedFusion(nn.Module):

    def __init__(self, feature_channels=FEATURE_CHANNELS):
        super().__init__()

        self.theta_conv = nn.Conv2d(
            3,
            feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )

    def forward(self, x_mid, original_x, f_spa, f_fre):

        hfe_mid = high_frequency_extraction(x_mid)

        hfe_input = high_frequency_extraction(original_x)

        abs_difference = torch.abs(hfe_mid - hfe_input)

        theta = self.theta_conv(abs_difference)

        theta = torch.sigmoid(theta)

        if theta.shape[-2:] != f_spa.shape[-2:]:
            theta = F.interpolate(
                theta,
                size=f_spa.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = theta * f_spa + (1.0 - theta) * f_fre

        return (fused, theta, hfe_mid, hfe_input)

class Stage2Decoder(nn.Module):

    def __init__(self, channels=FEATURE_CHANNELS):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        self.output = nn.Conv2d(
            channels,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x, output_size):

        x = self.conv1(x)
        x = self.conv2(x)

        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)

        residual = self.output(x)

        return residual

# Stage 2

class Stage2DFMSRN(nn.Module):

    def __init__(self, feature_channels=FEATURE_CHANNELS):
        super().__init__()

        self.spatial_domain = SpatialDomainModule(channels=feature_channels)

        self.fourier_domain = FourierDomainModule(channels=feature_channels)

        self.hfe_fusion = HFEBasedFusion(feature_channels=feature_channels)

        self.decoder = Stage2Decoder(channels=feature_channels)

    def forward(self, x_mid, original_x):

        f_spa = self.spatial_domain(x_mid)

        f_fre = self.fourier_domain(x_mid)

        (
            f_fused,
            theta,
            hfe_mid,
            hfe_input,
        ) = self.hfe_fusion(
            x_mid=x_mid,
            original_x=original_x,
            f_spa=f_spa,
            f_fre=f_fre,
        )

        # Decoder

        residual = self.decoder(f_fused, output_size=x_mid.shape[-2:])

        # Residual restoration around X_mid.
        x_fin = x_mid + residual

        return (x_fin, theta, f_spa, f_fre)
