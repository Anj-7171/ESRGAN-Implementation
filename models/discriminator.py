import torch.nn as nn


def conv_block(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_features=64, input_size=128):
        super(Discriminator, self).__init__()
        nf = base_features

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(nf,      nf,      stride=2),
            conv_block(nf,      nf * 2,  stride=1),
            conv_block(nf * 2,  nf * 2,  stride=2),
            conv_block(nf * 2,  nf * 4,  stride=1),
            conv_block(nf * 4,  nf * 4,  stride=2),
            conv_block(nf * 4,  nf * 8,  stride=1),
            conv_block(nf * 8,  nf * 8,  stride=2),
        )

        spatial = input_size // 16
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((spatial, spatial)),
            nn.Flatten(),
            nn.Linear(nf * 8 * spatial * spatial, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))