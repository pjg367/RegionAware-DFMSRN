import torch.nn as nn
from .stage1 import Stage1DFMSRN
from .stage2 import Stage2DFMSRN

class DFMSRN(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = Stage1DFMSRN()

        self.stage2 = Stage2DFMSRN()

    def forward(self, x):

        # Stage 1

        (x_mid, predicted_amplitude, raw_amplitude) = self.stage1(x)

        # Stage 2

        # SD / FD input = X_mid

        # HFE comparison = original X

        (x_fin, theta, f_spa, f_fre) = self.stage2(x_mid=x_mid, original_x=x)

        return {
            "x_mid":
                x_mid,
            "x_fin":
                x_fin,
            "predicted_amplitude":
                predicted_amplitude,
            "raw_amplitude":
                raw_amplitude,
            "theta":
                theta,
            "f_spa":
                f_spa,
            "f_fre":
                f_fre,
        }
