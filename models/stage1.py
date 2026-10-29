import torch
import torch.nn as nn
import torch.nn.functional as F

# Config

FFT_NORM = "ortho"

def fft_decompose(image):
    """
    image:
        B, C, H, W

    return:
        amplitude
        phase
    """

    fourier = torch.fft.fft2(image, dim=(-2, -1), norm=FFT_NORM)

    amplitude = torch.abs(fourier)
    phase = torch.angle(fourier)

    return amplitude, phase

def ifft_reconstruct(amplitude, phase):
    """
    Restore spatial image from amplitude + phase.
    """

    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)

    complex_fourier = torch.complex(real, imag)

    image = torch.fft.ifft2(complex_fourier, dim=(-2, -1), norm=FFT_NORM)

    return image.real

# Multi-resolution Fourier Amplitude

def get_multires_amplitudes(image):

    h, w = image.shape[-2:]

    amp_full, phase_full = fft_decompose(image)

    # 1/2

    image_half = F.interpolate(
        image,
        size=(h // 2, w // 2),
        mode="bilinear",
        align_corners=False,
    )

    amp_half, _ = fft_decompose(image_half)

    # 1/4

    image_quarter = F.interpolate(
        image,
        size=(h // 4, w // 4),
        mode="bilinear",
        align_corners=False,
    )

    amp_quarter, _ = fft_decompose(image_quarter)

    # 1/8

    image_eighth = F.interpolate(
        image,
        size=(h // 8, w // 8),
        mode="bilinear",
        align_corners=False,
    )

    amp_eighth, _ = fft_decompose(image_eighth)

    return (amp_full, amp_half, amp_quarter, amp_eighth, phase_full)

class InitialBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.InstanceNorm2d(
                32,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(
                out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(
                out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

    def forward(self, x):
        return self.block(x)

class AmplitudeAffineNetwork(nn.Module):

    def __init__(self, output_channels):
        super().__init__()

        hidden_channels = 32

        self.shared = nn.Sequential(
            nn.Conv2d(
                3,
                hidden_channels,
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

        self.alpha = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.beta = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, amplitude):

        feature = self.shared(amplitude)

        alpha = self.alpha(feature)
        beta = self.beta(feature)

        return alpha, beta

# Fourier Spatially Adaptive Normalization

class FourierSAN(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(channels, affine=False)

        self.affine_network = AmplitudeAffineNetwork(channels)

    def forward(self, feature, amplitude):

        if feature.shape[-2:] != amplitude.shape[-2:]:
            raise RuntimeError(
                "\nSAN resolution mismatch\n"
                f"Feature   : {feature.shape}\n"
                f"Amplitude : {amplitude.shape}"
            )

        normalized = self.batch_norm(feature)

        (alpha, beta) = self.affine_network(amplitude)

        output = alpha * normalized + beta

        return output

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        combined_channels = in_channels + skip_channels

        self.san1 = FourierSAN(combined_channels)

        self.conv1 = nn.Conv2d(
            combined_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.san2 = FourierSAN(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, skip, amplitude):

        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat((x, skip), dim=1)

        x = self.san1(x, amplitude)

        x = self.activation(x)

        x = self.conv1(x)

        x = self.san2(x, amplitude)

        x = self.activation(x)

        x = self.conv2(x)

        return x

class OutputBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(
                32,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
            nn.Conv2d(
                32,
                3,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.block(x)

class FSANNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.initial = InitialBlock()

        # Encoder

        self.encoder1 = EncoderBlock(32, 64)

        self.encoder2 = EncoderBlock(64, 128)

        self.encoder3 = EncoderBlock(128, 256)

        self.encoder4 = EncoderBlock(256, 512)

        # Decoder

        self.decoder1 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.decoder2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.decoder3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        self.decoder4 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)

        self.output = OutputBlock()

    def forward(self, amp_full, amp_half, amp_quarter, amp_eighth):

        # Encoder

        x0 = self.initial(amp_full)

        x1 = self.encoder1(x0)

        x2 = self.encoder2(x1)

        x3 = self.encoder3(x2)

        x4 = self.encoder4(x3)

        # Decoder

        d1 = self.decoder1(x4, x3, amp_eighth)

        d2 = self.decoder2(d1, x2, amp_quarter)

        d3 = self.decoder3(d2, x1, amp_half)

        d4 = self.decoder4(d3, x0, amp_full)

        raw_amplitude = self.output(d4)

        # Fourier amplitude must be positive.

        predicted_amplitude = F.softplus(raw_amplitude)

        return (predicted_amplitude, raw_amplitude)

# Stage 1

class Stage1DFMSRN(nn.Module):

    def __init__(self):
        super().__init__()

        self.f_sannet = FSANNet()

    def forward(self, image):

        (
            amp_full,
            amp_half,
            amp_quarter,
            amp_eighth,
            phase_full,
        ) = get_multires_amplitudes(
            image
        )

        (
            predicted_amplitude,
            raw_amplitude,
        ) = self.f_sannet(
            amp_full,
            amp_half,
            amp_quarter,
            amp_eighth,
        )

        # Predicted amplitude
        # original input phase

        x_mid = ifft_reconstruct(predicted_amplitude, phase_full)

        return (x_mid, predicted_amplitude, raw_amplitude)
