import torch
import torch.nn as nn
import torch.nn.functional as F


# U-Net model
# convolution class
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
        self.cov1 = Conv(3, 64)
        self.encoder1 = DownSampling(64)
        self.cov2 = Conv(64, 128)
        self.encoder2 = DownSampling(128)
        self.cov3 = Conv(128, 256)
        self.encoder3 = DownSampling(256)
        self.cov4 = Conv(256, 512)
        self.encoder4 = DownSampling(512)
        self.cov5 = Conv(512, 1024)

        # 4 times up sampling
        self.decoder1 = UpSampling(1024)
        self.cov6 = Conv(1024, 512)
        self.decoder2 = UpSampling(512)
        self.cov7 = Conv(512, 256)
        self.decoder3 = UpSampling(256)
        self.cov8 = Conv(256, 128)
        self.decoder4 = UpSampling(128)
        self.cov9= Conv(128, 64)

        # self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        # down sampling
        result1 = self.cov1(x)
        result2 = self.cov2(self.encoder1(result1))
        result3 = self.cov3(self.encoder2(result2))
        result4 = self.cov4(self.encoder3(result3))
        result5 = self.cov5(self.encoder4(result4))

        # up sampling
        result6 = self.cov6(self.decoder1(result5, result4))
        result7 = self.cov7(self.decoder2(result6, result3))
        result8 = self.cov8(self.decoder3(result7, result2))
        result9 = self.cov9(self.decoder4(result8, result1))

        # output
        return self.pred(result9)  # self.Th(self.pred(O4))
