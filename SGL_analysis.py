import os
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask_1208/0.6/analysis/feature_stats.csv"

csv_path = "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask_1208_foreground/0.6/analysis/feature_stats.csv"
assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"

iters = []
norm_mean = []
norm_std  = []
ent_mean  = []
ent_std   = []

# CSV 읽기 (헤더 스킵, 행 번호를 iteration으로 사용)
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        iters.append(i + 1)  # iteration index 시작을 1로
        # before 통계를 사용 (요청 내용에 맞게)
        norm_mean.append(float(row["norm_before_mean"]))
        norm_std.append(float(row["norm_before_std"]))
        ent_mean.append(float(row["ent_before_mean"]))
        ent_std.append(float(row["ent_before_std"]))

iters = np.array(iters)
norm_mean = np.array(norm_mean)
norm_std  = np.array(norm_std)
ent_mean  = np.array(ent_mean)
ent_std   = np.array(ent_std)

# 플롯
plt.figure(figsize=(12, 5))

# (1) Norm
ax1 = plt.subplot(1, 2, 1)
ax1.plot(iters, norm_mean, label="norm_before_mean", color="C0")
ax1.fill_between(iters, norm_mean - norm_std, norm_mean + norm_std, color="C0", alpha=0.2, label="± std")
ax1.set_title("Feature L2 Norm (per iteration)")
ax1.set_xlabel("iteration")
ax1.set_ylabel("mean ± std")
ax1.grid(True, alpha=0.2)
ax1.legend()

# (2) Entropy
ax2 = plt.subplot(1, 2, 2)
ax2.plot(iters, ent_mean, label="entropy_before_mean", color="C1")
ax2.fill_between(iters, ent_mean - ent_std, ent_mean + ent_std, color="C1", alpha=0.2, label="± std")
ax2.set_title("Feature Entropy (per iteration)")
ax2.set_xlabel("iteration")
ax2.set_ylabel("mean ± std")
ax2.grid(True, alpha=0.2)
ax2.legend()

plt.tight_layout()
out_png = os.path.join(os.path.dirname(csv_path), "feature_iter_plots_2.png")
plt.savefig(out_png, dpi=150)
print(f"Saved plot: {out_png}")