import os, argparse, logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.generator     import RRDBNet
from models.discriminator import Discriminator
from models.losses        import (PixelLoss, ESRGANGeneratorLoss,
                                  RelativisticAdversarialLoss)
from data.dataset         import MRISliceDataset
from utils.metrics        import compute_psnr, compute_ssim
from utils.checkpoint     import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",        default="psnr", choices=["psnr", "gan"])
    p.add_argument("--data_root",    required=True)
    p.add_argument("--scale",        type=int, default=4)
    p.add_argument("--hr_patch",     type=int, default=128)
    p.add_argument("--num_blocks",   type=int, default=23)
    p.add_argument("--in_channels",  type=int, default=1)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--epochs",       type=int, default=None)
    p.add_argument("--lr_g",         type=float, default=1e-4)
    p.add_argument("--lr_d",         type=float, default=1e-4)
    p.add_argument("--pretrained_g", default=None)
    p.add_argument("--resume",       default=None)
    p.add_argument("--save_dir",     default="checkpoints")
    p.add_argument("--log_dir",      default="runs")
    p.add_argument("--log_freq",     type=int, default=100)
    p.add_argument("--val_freq",     type=int, default=1)
    p.add_argument("--save_freq",    type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def validate(G, loader, device):
    G.eval()
    psnr, ssim, n = 0., 0., 0
    for batch in loader:
        sr = G(batch["lr"].to(device)).clamp(0, 1)
        hr = batch["hr"].to(device)
        for i in range(sr.shape[0]):
            psnr += compute_psnr(sr[i], hr[i])
            ssim += compute_ssim(sr[i], hr[i])
            n += 1
    return psnr / n, ssim / n


def get_data(args):
    train = MRISliceDataset(args.data_root, "train", args.hr_patch, args.scale)
    val   = MRISliceDataset(args.data_root, "val",   args.hr_patch, args.scale,
                            augment=False)
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    return (DataLoader(train, args.batch_size, shuffle=True,  **kw),
            DataLoader(val,   args.batch_size, shuffle=False, **kw))


# ── Phase 1: PSNR pretraining ──────────────────────────────────────────────

def train_psnr(args, device):
    log.info("=== Phase 1: PSNR Pretraining ===")
    epochs = args.epochs or 100
    G = RRDBNet(args.in_channels, args.in_channels,
                num_blocks=args.num_blocks, scale_factor=args.scale).to(device)
    opt  = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-7)
    loss_fn = PixelLoss().to(device)

    train_dl, val_dl = get_data(args)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, "psnr"))
    start, best = (load_checkpoint(args.resume, G, opt) if args.resume
                   else (0, 0.0))

    for epoch in range(start, epochs):
        G.train()
        for step, batch in enumerate(train_dl):
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            loss = loss_fn(G(lr), hr)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % args.log_freq == 0:
                gs = epoch * len(train_dl) + step
                writer.add_scalar("train/l1", loss.item(), gs)
                log.info(f"E{epoch+1}/{epochs} S{step} l1={loss.item():.4f}")
        sched.step()

        if (epoch + 1) % args.val_freq == 0:
            psnr, ssim = validate(G, val_dl, device)
            writer.add_scalar("val/PSNR", psnr, epoch)
            writer.add_scalar("val/SSIM", ssim, epoch)
            log.info(f"[VAL] PSNR={psnr:.2f} SSIM={ssim:.4f}")
            if psnr > best:
                best = psnr
                save_checkpoint(G, opt, epoch, psnr,
                                f"{args.save_dir}/psnr_best.pth")

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(G, opt, epoch, 0,
                            f"{args.save_dir}/psnr_e{epoch+1}.pth")
    writer.close()


# ── Phase 2: GAN training ──────────────────────────────────────────────────

def train_gan(args, device):
    log.info("=== Phase 2: GAN Training ===")
    epochs = args.epochs or 400
    G = RRDBNet(args.in_channels, args.in_channels,
                num_blocks=args.num_blocks, scale_factor=args.scale).to(device)
    D = Discriminator(args.in_channels, input_size=args.hr_patch).to(device)

    if args.pretrained_g:
        load_checkpoint(args.pretrained_g, G)
        log.info(f"Loaded pretrained G: {args.pretrained_g}")

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    ms = [int(epochs * r) for r in [0.5, 0.75, 0.875, 0.9375]]
    sched_G = torch.optim.lr_scheduler.MultiStepLR(opt_G, ms, gamma=0.5)
    sched_D = torch.optim.lr_scheduler.MultiStepLR(opt_D, ms, gamma=0.5)

    g_loss = ESRGANGeneratorLoss().to(device)
    d_loss = RelativisticAdversarialLoss().to(device)

    train_dl, val_dl = get_data(args)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, "gan"))
    best = 0.0

    for epoch in range(epochs):
        G.train(); D.train()
        for step, batch in enumerate(train_dl):
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            gs = epoch * len(train_dl) + step

            # D step
            with torch.no_grad(): sr = G(lr)
            rl, fl = D(hr), D(sr.detach())
            ld = d_loss.discriminator_loss(rl, fl)
            opt_D.zero_grad(); ld.backward(); opt_D.step()

            # G step
            sr = G(lr)
            rl, fl = D(hr).detach(), D(sr)
            lg, info = g_loss(sr, hr, rl, fl)
            opt_G.zero_grad(); lg.backward(); opt_G.step()

            if step % args.log_freq == 0:
                writer.add_scalar("train/G", lg.item(), gs)
                writer.add_scalar("train/D", ld.item(), gs)
                log.info(f"E{epoch+1}/{epochs} S{step} "
                         f"G={lg.item():.4f} D={ld.item():.4f} "
                         f"pix={info['pixel']:.4f} "
                         f"perc={info['perceptual']:.4f}")

        sched_G.step(); sched_D.step()

        if (epoch + 1) % args.val_freq == 0:
            psnr, ssim = validate(G, val_dl, device)
            writer.add_scalar("val/PSNR", psnr, epoch)
            writer.add_scalar("val/SSIM", ssim, epoch)
            log.info(f"[VAL] PSNR={psnr:.2f} SSIM={ssim:.4f}")
            if psnr > best:
                best = psnr
                save_checkpoint(G, opt_G, epoch, psnr,
                                f"{args.save_dir}/esrgan_best.pth", D, opt_D)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(G, opt_G, epoch, 0,
                            f"{args.save_dir}/esrgan_e{epoch+1}.pth", D, opt_D)
    writer.close()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    (train_psnr if args.phase == "psnr" else train_gan)(args, device)


if __name__ == "__main__":
    main()