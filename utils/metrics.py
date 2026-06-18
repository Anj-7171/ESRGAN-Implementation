import math, torch, torch.nn.functional as F


def compute_psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr).item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(max_val ** 2 / mse)


def compute_ssim(sr, hr, win=11, sigma=1.5, C1=1e-4, C2=9e-4):
    sr, hr = sr.unsqueeze(0), hr.unsqueeze(0)
    coords = torch.arange(win, dtype=torch.float32, device=sr.device) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    k = g.outer(g).unsqueeze(0).unsqueeze(0).expand(sr.shape[1], 1, win, win)
    pad = win // 2

    mu1 = F.conv2d(sr, k, padding=pad, groups=sr.shape[1])
    mu2 = F.conv2d(hr, k, padding=pad, groups=hr.shape[1])
    s12 = F.conv2d(sr * hr, k, padding=pad, groups=sr.shape[1]) - mu1 * mu2
    s1  = F.conv2d(sr * sr, k, padding=pad, groups=sr.shape[1]) - mu1 ** 2
    s2  = F.conv2d(hr * hr, k, padding=pad, groups=hr.shape[1]) - mu2 ** 2

    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
            ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))).mean().item()