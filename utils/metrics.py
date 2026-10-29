import torch

def calculate_psnr(prediction, target):

    prediction = torch.clamp(prediction, 0.0, 1.0)

    target = torch.clamp(target, 0.0, 1.0)

    mse = torch.mean((prediction - target) ** 2)

    if mse.item() <= 1e-12:

        return 100.0

    psnr = 10.0 * torch.log10(1.0 / mse)

    return psnr.item()

class AverageMeter:

    def __init__(self):

        self.sum = 0.0

        self.count = 0

    def update(self, value, n=1):

        self.sum += float(value) * n

        self.count += n

    @property

    def avg(self):

        if self.count == 0:

            return 0.0

        return self.sum / self.count
