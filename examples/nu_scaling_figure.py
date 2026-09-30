"""Required integration time vs dither frequency nu (dither power -25 dB re data).

T ∝ 1/nu^2 from sigma = 1/(2 pi nu sqrt(p B_d T)). Anchors: full-band skew burst
3.6 s at 90 MHz (0.1 ps), 5-GHz-slice phase sweep 1.6 s at 90 MHz (0.1 rad);
block-gain (ZOH) implementation at 90 MHz costs sinc(90/277.8)^-2 = 1.43.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

f_cn = font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
nu = np.logspace(np.log10(50e6), np.log10(1.2e9), 200)
T_skew = 3.6 * (90e6 / nu) ** 2
T_phase = 1.6 * (90e6 / nu) ** 2
zoh = 1 / np.sinc(90e6 / 277.8e6) ** 2

C1, C2, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(9, 5.6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.loglog(nu / 1e6, T_skew, color=C1, lw=2, label="skew 到 0.1 ps（全带 burst）")
ax.loglog(nu / 1e6, T_phase, color=C2, lw=2, label="相频到 0.1 rad（5 GHz 片扫频）")

pts = [(90, "块级增益 ZOH", zoh), (277.8, "邻 bin MAC (m=1)", 1.0), (555.6, "隔 bin MAC (m=2)", 1.0)]
for x, name, k in pts:
    ys = 3.6 * (90 / x) ** 2 * k
    yp = 1.6 * (90 / x) ** 2 * k
    ax.plot([x], [ys], "o", color=C1, ms=9, mec="#fcfcfb", mew=2)
    ax.plot([x], [yp], "o", color=C2, ms=9, mec="#fcfcfb", mew=2)
    ax.annotate(f"{ys:.2f} s", (x, ys), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=10)
    ax.annotate(f"{yp:.2f} s", (x, yp), xytext=(8, -14), textcoords="offset points", color=INK, fontsize=10)
    ax.axvline(x, color="#d0cfc9", lw=0.8, ls=":")
    ax.text(x, 0.011, f"ν = {x:g} MHz\n{name}\nADC ≥ {2.2 * x / 1000:.2f} GS/s",
            ha="center", va="bottom", fontsize=9, color=MUTED, fontproperties=f_cn,
            bbox=dict(fc="#fcfcfb", ec="none", pad=1))
# ZOH-free 90 MHz reference (pure physics)
ax.plot([90], [3.6], "o", mfc="none", mec=C1, ms=9, mew=1.5)
ax.annotate("3.6 s（无 ZOH 损失）", (90, 3.6), xytext=(-8, -16), textcoords="offset points",
            ha="right", color=MUTED, fontsize=9, fontproperties=f_cn)

ax.set_xlim(50, 1200); ax.set_ylim(0.01, 30)
ax.set_xlabel("导频频率 ν (MHz)", fontproperties=f_cn, fontsize=11)
ax.set_ylabel("所需总积分时间 (s)，两支路合计", fontproperties=f_cn, fontsize=11)
ax.set_title("积分时间随导频频率的变化：T ∝ 1/ν²，dither 功率 −25 dB，ν=90 MHz 处对齐仿真",
             fontproperties=f_cn, fontsize=12, color=INK)
ax.grid(True, which="major", color="#e6e5e0", lw=0.8)
ax.grid(True, which="minor", color="#f0efea", lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(prop=f_cn, fontsize=10, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("docs/figures/nu_scaling.png", dpi=150, facecolor=fig.get_facecolor())
print("saved docs/figures/nu_scaling.png")
