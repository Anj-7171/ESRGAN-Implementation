from pathlib import Path
import csv
import subprocess

DATA_ROOT = "data"

EPOCHS = [5, 10, 15, 20]

results = []

Path("results_esrgan").mkdir(exist_ok=True)

for epoch in EPOCHS:

    checkpoint = f"checkpoints/esrgan_e{epoch}.pth"

    if not Path(checkpoint).exists():
        print(f"Skipping epoch {epoch}: checkpoint not found")
        continue

    output_dir = f"results_esrgan/e{epoch}"

    cmd = [
        "python",
        "evaluate.py",
        "--checkpoint", checkpoint,
        "--data_root", DATA_ROOT,
        "--output_dir", output_dir
    ]

    print(f"\nEvaluating Epoch {epoch}")
    subprocess.run(cmd)

    metrics_file = Path(output_dir) / "metrics.txt"

    if metrics_file.exists():

        psnr = None
        ssim = None

        with open(metrics_file, "r") as f:
            for line in f:
                if line.startswith("PSNR"):
                    psnr = float(line.split(":")[1].strip())
                elif line.startswith("SSIM"):
                    ssim = float(line.split(":")[1].strip())

        results.append([epoch, psnr, ssim])

# Evaluate best checkpoint
best_checkpoint = "checkpoints/esrgan_best.pth"

if Path(best_checkpoint).exists():

    output_dir = "results_esrgan/best"

    cmd = [
        "python",
        "evaluate.py",
        "--checkpoint", best_checkpoint,
        "--data_root", DATA_ROOT,
        "--output_dir", output_dir
    ]

    print("\nEvaluating Best Model")
    subprocess.run(cmd)

csv_path = Path("results_esrgan") / "summary_metrics.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["Epoch", "PSNR", "SSIM"])

    for row in results:
        writer.writerow(row)

print("\nSaved summary:")
print(csv_path)