import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class MRISliceDataset(Dataset):
    def __init__(self, root, split="train", hr_patch_size=128, scale=4, augment=True):
        self.hr = hr_patch_size
        self.lr = hr_patch_size // scale
        self.augment = augment and (split == "train")

        root = Path(root) / split
        self.paths = sorted(root.glob("*.png")) or sorted(root.glob("*.jpg"))
        assert self.paths, f"No images in {root}"
        print(f"[{split}] {len(self.paths)} slices")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = TF.to_tensor(Image.open(self.paths[idx]).convert("L"))
        _, H, W = img.shape

        if H < self.hr or W < self.hr:
            img = TF.resize(img, [self.hr, self.hr],
                            interpolation=TF.InterpolationMode.BICUBIC)
            H = W = self.hr

        if self.augment:
            top = random.randint(0, H - self.hr)
            left = random.randint(0, W - self.hr)
        else:
            top = (H - self.hr) // 2
            left = (W - self.hr) // 2

        hr = TF.crop(img, top, left, self.hr, self.hr)
        lr = TF.resize(hr, [self.lr, self.lr],
                       interpolation=TF.InterpolationMode.BICUBIC)

        if self.augment:
            if random.random() > 0.5: hr, lr = TF.hflip(hr), TF.hflip(lr)
            if random.random() > 0.5: hr, lr = TF.vflip(hr), TF.vflip(lr)
            k = random.randint(0, 3)
            hr = torch.rot90(hr, k, [1, 2])
            lr = torch.rot90(lr, k, [1, 2])

        return {"lr": lr, "hr": hr}


class FastMRIDataset(Dataset):
    def __init__(self, root, split="train", hr_patch_size=128, scale=4,
                 max_slices=20):
        import h5py
        self.h5py = h5py
        self.hr, self.lr = hr_patch_size, hr_patch_size // scale
        self.augment = (split == "train")
        self.samples = []

        for fp in sorted((Path(root) / split).glob("*.h5")):
            with h5py.File(fp) as f:
                key = "reconstruction_rss" if "reconstruction_rss" in f else "kspace"
                for i in range(min(f[key].shape[0], max_slices)):
                    self.samples.append((fp, i, key))

        print(f"[{split}] FastMRI: {len(self.samples)} slices")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        fp, i, key = self.samples[idx]
        with self.h5py.File(fp) as f:
            sl = f[key][i]
            if sl.ndim == 3:
                sl = np.sqrt((np.abs(sl) ** 2).sum(0))
            mn, mx = sl.min(), sl.max()
            sl = ((sl - mn) / (mx - mn + 1e-8)).astype(np.float32)

        hr = torch.from_numpy(sl).unsqueeze(0)
        _, H, W = hr.shape
        if H < self.hr or W < self.hr:
            hr = TF.resize(hr, [self.hr, self.hr],
                           interpolation=TF.InterpolationMode.BICUBIC)
            H = W = self.hr

        if self.augment:
            top = random.randint(0, H - self.hr)
            left = random.randint(0, W - self.hr)
        else:
            top = (H - self.hr) // 2
            left = (W - self.hr) // 2
        hr = TF.crop(hr, top, left, self.hr, self.hr)
        lr = TF.resize(hr, [self.lr, self.lr],
                       interpolation=TF.InterpolationMode.BICUBIC)
        return {"lr": lr, "hr": hr}