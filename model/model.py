import torch
import torch.nn as nn
from torchinfo import summary


class CNNBlock(nn.Module):
    def __init__(self, in_channles, out_channels, stride=2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channles,
                out_channels,
                kernel_size=4,
                stride=stride,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


class UNETBlock(nn.Module):
    def __init__(
        self, in_channles, out_channles, down, act="relu", use_dropout=False
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channles,
                out_channles,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            if down
            else nn.ConvTranspose2d(
                in_channles,
                out_channles,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channles),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels, features=64):
        super().__init__()
        # Input images for plaque dataset are of size 512x672. In the initial
        # down layer, we want to modify the size with padding such that the
        # following down sampling layers are easier to do. Therefore, we pad
        # left and right by a total of 768-672=96 pixels. Why 768? because it
        # results in 2x3 at bottleneck by consecutive 2x downsampling. In the
        # end, we remove the boundaries to get the original size.
        self.initial_down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )  # 256 x 384

        # Encoder
        self.down1 = UNETBlock(
            features, features * 2, down=True, act="leaky", use_dropout=False
        )  # 128 x 192
        self.down2 = UNETBlock(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )  # 64 x 96
        self.down3 = UNETBlock(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )  # 32 x 48
        self.down4 = UNETBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # 16 x 24
        self.down5 = UNETBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # 8 x 12
        self.down6 = UNETBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # 4 X 6
        self.down7 = UNETBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )  # 2 X 6

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features * 8,
                features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
        )  # 2 x 3

        # Decoder
        self.up1 = UNETBlock(
            features * 8, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up2 = UNETBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = UNETBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = UNETBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = UNETBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up6 = UNETBlock(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up7 = UNETBlock(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up8 = UNETBlock(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False
        )

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, in_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        d8 = self.down7(d7)

        bottleneck = self.bottleneck(d8)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d8], 1))
        up3 = self.up3(torch.cat([up2, d7], 1))
        up4 = self.up4(torch.cat([up3, d6], 1))
        up5 = self.up5(torch.cat([up4, d5], 1))
        up6 = self.up6(torch.cat([up5, d4], 1))
        up7 = self.up7(torch.cat([up6, d3], 1))
        up8 = self.up8(torch.cat([up7, d2], 1))

        return self.final_up(torch.cat([up8, d1], 1))


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(device)

    def test_disc():
        x = torch.randn((1, 1, 512, 512))
        y = torch.randn((1, 1, 512, 512))
        model = Discriminator(in_channels=1)
        summary(model, input_size=[(1, 1, 512, 512), (1, 1, 512, 512)])
        pred = model(x, y)
        print(pred.shape)

    def test_gen():
        x = torch.randn((1, 1, 512, 512))
        model = Generator(in_channels=1, features=64)
        summary(model, input_size=(1, 1, 512, 512))
        preds = model(x)
        print(preds.shape)

    # test_disc()
    test_gen()
