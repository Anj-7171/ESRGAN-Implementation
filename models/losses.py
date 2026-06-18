import torch
import torch.nn as nn
import torchvision.models as models


class PixelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        return self.l1(sr, hr)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=18):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:layer_index]).eval()
        for p in self.features.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.features(x)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.l1  = nn.L1Loss()

    def forward(self, sr, hr):
        return self.l1(self.vgg(sr), self.vgg(hr).detach())


class RelativisticAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def generator_loss(self, real_logits, fake_logits):
        return (
            self.bce(fake_logits - real_logits.mean(), torch.ones_like(fake_logits)) +
            self.bce(real_logits - fake_logits.mean(), torch.zeros_like(real_logits))
        ) / 2

    def discriminator_loss(self, real_logits, fake_logits):
        return (
            self.bce(real_logits - fake_logits.mean(), torch.ones_like(real_logits)) +
            self.bce(fake_logits - real_logits.mean(), torch.zeros_like(fake_logits))
        ) / 2


class ESRGANGeneratorLoss(nn.Module):
    def __init__(self, lambda_pixel=0.01, lambda_percep=1.0, lambda_adv=0.005):
        super().__init__()
        self.pixel_loss  = PixelLoss()
        self.percep_loss = PerceptualLoss()
        self.adv_loss    = RelativisticAdversarialLoss()
        self.lp, self.lc, self.la = lambda_pixel, lambda_percep, lambda_adv

    def forward(self, sr, hr, real_logits, fake_logits):
        lp = self.pixel_loss(sr, hr)
        lc = self.percep_loss(sr, hr)
        la = self.adv_loss.generator_loss(real_logits, fake_logits)
        total = self.lp * lp + self.lc * lc + self.la * la
        return total, {"pixel": lp.item(), "perceptual": lc.item(), "adversarial": la.item()}