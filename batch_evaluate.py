from pathlib import Path
import csv
import subprocess

DATA_ROOT = "data"

EPOCHS = [20, 40, 60, 80, 100]

results = []

for epoch in EPOCHS:

    checkpoint = f"checkpoints/psnr_e{epoch}.pth"

    if not Path(checkpoint).exists():
        print(f"Skipping epoch {epoch}: checkpoint not found")
        continue

    output_dir = f"results/e{epoch}"

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

csv_path = Path("results") / "summary_metrics.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["Epoch", "PSNR", "SSIM"])

    for row in results:
        writer.writerow(row)

print("\nSaved summary:")
print(csv_path)