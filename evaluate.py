import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF

from models.generator import RRDBNet
from data.dataset import MRISliceDataset
from utils.metrics import compute_psnr, compute_ssim
from utils.checkpoint import load_checkpoint


def save_image(tensor, path):
    arr = (tensor.squeeze().cpu().numpy() * 255)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def create_comparison(lr, sr, hr, save_path):
    lr_img = TF.to_pil_image(lr)
    sr_img = TF.to_pil_image(sr)
    hr_img = TF.to_pil_image(hr)

    w, h = hr_img.size

    lr_img = lr_img.resize((w, h), Image.BICUBIC)

    canvas = Image.new("L", (w * 3, h))

    canvas.paste(lr_img, (0, 0))
    canvas.paste(sr_img, (w, 0))
    canvas.paste(hr_img, (2 * w, 0))

    canvas.save(save_path)


@torch.no_grad()
def evaluate(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = RRDBNet(
        in_channels=1,
        out_channels=1,
        num_blocks=args.num_blocks,
        scale_factor=args.scale
    ).to(device)

    load_checkpoint(args.checkpoint, model)

    model.eval()

    dataset = MRISliceDataset(
        args.data_root,
        split="val",
        hr_patch_size=args.hr_patch,
        scale=args.scale,
        augment=False
    )

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_psnr = 0
    total_ssim = 0

    sample_saved = False

    for idx in range(len(dataset)):

        sample = dataset[idx]

        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"]

        sr = model(lr).clamp(0, 1)

        sr_cpu = sr.squeeze(0).cpu()

        total_psnr += compute_psnr(sr_cpu, hr)
        total_ssim += compute_ssim(sr_cpu, hr)

        if not sample_saved:

            save_image(lr.squeeze(0),
                       results_dir / "lr.png")

            save_image(sr_cpu,
                       results_dir / "sr.png")

            save_image(hr,
                       results_dir / "hr.png")

            create_comparison(
                lr.squeeze(0),
                sr_cpu,
                hr,
                results_dir / "comparison.png"
            )

            sample_saved = True

    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)

    print("\n==========================")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"PSNR       : {avg_psnr:.4f}")
    print(f"SSIM       : {avg_ssim:.4f}")
    print("==========================\n")

    with open(results_dir / "metrics.txt", "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4
    )

    parser.add_argument(
        "--hr_patch",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num_blocks",
        type=int,
        default=23
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())