import torch
import torch.nn as nn
import torch.nn.functional as F


# U-Net model
# convolutiuon class
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # prevent overfitting
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # prevent overfitting
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# down sampling module
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 2X downsampling using convolution with the same number of channels
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.Down(x)


# up sampling module
class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        # Feature map size is expanded by 2 times and the number of channels is halved
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # Downsampling using neighborhood interpolation
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # concat, the current upsampling, and the previous downsampling process
        return torch.cat((x, r), 1)


# U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 4 times down sampling
        self.C1 = Conv(3, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4 times up sampling
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        # self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        # down sampling
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # up sampling
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # output
        return self.pred(O4)  # self.Th(self.pred(O4))
