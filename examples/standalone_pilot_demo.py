"""乘性导频 S21 监测：自包含的原理演示（只依赖 numpy + matplotlib）

运行:  python standalone_pilot_demo.py          （4 核约 1.5 分钟，单核约 5 分钟）
输出:  standalone_pilot_demo.png + 终端里的数字

链路（每条 1 µs 记录都按 200 GS/s 逐样本仿真，数据每次新生成）:

  发端 DSP:   x_X(t) -> x_X(t)·[1 + ε·cos(2πνt)]      只对一条支路、或只对一段频谱加
  发端链路:   Y_X(f) = X_X(f)·H_X(f)                     H 含公共二次相频 + Q 路 skew
  近端 PD:    p(t) = y_I² + y_Q²                          平方律
  前端+ADC:   z = 低通 M(100 MHz) -> 200 MS/s 抽样 -> 加热噪声
  监测端:     B = Σ w[n]·z[n]·e^{-j2πν n/f_adc}            数字锁相，只看 ν 这一个频点

原理（见 docs/dither_principle_derivation.pdf）:
  每个频率 f 的数据分量 X(f)H(f) 和它的副本 (ε/2)X(f)H(f+ν) 在 PD 里互拍，
  数据相位抵消，剩下 θ(f+ν)−θ(f) = −2πν·τ_g(f)。所有 f 叠在 ν 一根线上，
  谱线相位 = 被调制频段的平均群时延。I/Q 相减得 skew；逐片扫描、逐片累加得相频。

调制深度故意设得很深（先不考虑对主数据流的代价），好让结果几十毫秒内就收敛。
"""
import os
import sys
import time
from multiprocessing import Pool, cpu_count, freeze_support

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 每个进程单线程，多进程并行更快
import numpy as np

# ------------------------------------------------------------------ 参数
fs = 200e9                 # 波形采样率 (200 GS/s)
N = 200_000                # 每条记录 1 µs
D_ADC = 1000               # 200 GS/s / 1000 = 200 MS/s
Nd = N // D_ADC            # 每记录 200 个 ADC 样本
fs_adc = fs / D_ADC
nu = 90e6                  # 导频频率：1 µs 内整 90 个周期，记录之间相位自动连续
BAUD = 100e9               # 100 GBaud 16QAM，2 sps，RRC 0.1
QUAD_RAD, F_EDGE = 6.0, 50e9   # 公共二次相频：50 GHz 处 −6 rad
SKEW_PS = 3.0              # Q 比 I 晚 3 ps（想试 0.3 ps 把时间加长 100 倍）
EPS_BURST = 0.5            # 全带 burst 的调制深度（测 skew）
EPS_SLICE = 1.2            # 分片扫描的调制深度（测相频）
T_BURST_MS = 1.5           # burst：每支路积分时间
T_SLICE_MS = 1.5           # 扫描：每片每支路积分时间
K = 5                      # 片数，5 × 10 GHz 铺满 0–50 GHz
SNR_ADC_DB = 30            # ADC 热噪声相对 PD 输出交流功率

f = np.fft.rfftfreq(N, 1 / fs)          # 记录内频率网格，1 MHz 一格
t = np.arange(N) / fs
Delta = F_EDGE / K
edges = np.arange(K + 1) * Delta


# ------------------------------------------------------------------ 链路各环节（全部在频域定义）
def rrc_spectrum(f, baud=BAUD, roll=0.1):
    """根升余弦成型滤波器的幅频"""
    fn = baud / 2
    a = np.ones_like(f)
    x = np.abs(f)
    m = (x > fn * (1 - roll)) & (x < fn * (1 + roll))
    a[m] = np.sqrt(0.5 * (1 + np.cos(np.pi / (2 * roll * fn) * (x[m] - fn * (1 - roll)))))
    a[x >= fn * (1 + roll)] = 0
    return a


def branch_response(f, f3db, gain, delay):
    """支路响应：平滑幅频 × 公共二次相频 × 纯延迟"""
    mag = gain / np.sqrt(1 + (f / f3db) ** 8)
    theta = -QUAD_RAD * (f / F_EDGE) ** 2 - 2 * np.pi * f * delay
    return mag * np.exp(1j * theta)


RRC = rrc_spectrum(f)
# 两路幅频形状取相同（只差增益），这样全带 burst 的理论值就是设定的 skew；
# 若两路滚降不同，burst 会多出一个加权偏置（见 PDF 6.1 节），分片差分不受影响
H_I = branch_response(f, 62e9, 1.00, 1.000e-9)
H_Q = branch_response(f, 62e9, 0.90, 1.000e-9 + SKEW_PS * 1e-12)
M = np.exp(-2j * np.pi * f * 20e-9) / np.sqrt(1 + (f / 100e6) ** 12)   # PD 前端 100 MHz，20 ns 延迟
LOCKIN = np.hanning(Nd) * np.exp(-2j * np.pi * nu * np.arange(Nd) / fs_adc)
LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])


def make_data(rng):
    """一条记录的 I、Q 驱动数据（频域）"""
    out = []
    for _ in range(2):
        up = np.zeros(N)
        up[::2] = rng.choice(LEVELS, N // 2)
        out.append(np.fft.rfft(up) * RRC)
    return out


def one_record(rng, branch, mask, eps, noise_std):
    """发端加导频 -> 支路 -> PD -> 前端 -> ADC -> 锁相。返回复数 B 和 ADC 谱"""
    X = make_data(rng)
    # ---- 发端：乘性导频。mask=None 为全带，否则只对 mask 内的频谱加
    x_part = np.fft.irfft(X[branch] * (1.0 if mask is None else mask), N)
    d = eps * x_part * np.cos(2 * np.pi * nu * t)
    X[branch] = X[branch] + np.fft.rfft(d)
    # ---- 支路响应，光场，PD 平方律
    yI = np.fft.irfft(X[0] * H_I, N)
    yQ = np.fft.irfft(X[1] * H_Q, N)
    p = yI ** 2 + yQ ** 2
    # ---- 前端低通 + 200 MS/s 抽样 + 热噪声
    z = np.fft.irfft(np.fft.rfft(p) * M, N)[::D_ADC]
    z = z + noise_std * rng.standard_normal(Nd)
    z = z - z.mean()
    # ---- 监测端：数字锁相，只取 ν 这一点
    B = np.sum(LOCKIN * z)
    spec = np.abs(np.fft.rfft(np.hanning(Nd) * z)) ** 2
    return B, spec


def worker(args):
    seed, n_rec, schedule, masks, eps, noise_std = args
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(schedule), complex)
    acc2 = np.zeros(len(schedule))
    cnt = np.zeros(len(schedule), int)
    spec = np.zeros(Nd // 2 + 1)
    for i in range(n_rec):
        s = i % len(schedule)
        branch, k = schedule[s]
        B, sp = one_record(rng, branch, masks[k], eps, noise_std)
        acc[s] += B
        acc2[s] += abs(B) ** 2
        cnt[s] += 1
        if s == 0:
            spec += sp
    return acc, acc2, cnt, spec


def run(schedule, masks, eps, noise_std, t_ms, procs, seed0):
    n_total = int(round(len(schedule) * t_ms * 1e-3 * fs / N))
    per = (n_total + procs - 1) // procs
    jobs = [(seed0 + j, per, schedule, masks, eps, noise_std) for j in range(procs)]
    t0 = time.time()
    if procs > 1:
        with Pool(procs) as pool:
            outs = pool.map(worker, jobs)
    else:
        outs = [worker(j) for j in jobs]
    acc = sum(o[0] for o in outs)
    acc2 = sum(o[1] for o in outs)
    cnt = sum(o[2] for o in outs)
    spec = sum(o[3] for o in outs) / max(1, cnt[0])
    B = acc / cnt
    sem = np.sqrt(np.maximum(acc2 / cnt - np.abs(B) ** 2, 0) / cnt / 2)   # 每个正交分量的标准误
    sig_phase = sem / np.abs(B)
    print(f"   {cnt.sum()} 条记录（{cnt.sum() * N / fs * 1e3:.1f} ms 信号）用时 {time.time() - t0:.0f} s")
    return B, sig_phase, spec


def main():
    procs = max(1, min(4, cpu_count()))
    print(f"ν = {nu / 1e6:.0f} MHz，ADC {fs_adc / 1e6:.0f} MS/s，PD 100 MHz，{procs} 个进程")

    # ADC 热噪声标定（不加导频时的 PD 输出交流功率）
    rng = np.random.default_rng(0)
    X = make_data(rng)
    z0 = np.fft.irfft(np.fft.rfft(np.fft.irfft(X[0] * H_I, N) ** 2 + np.fft.irfft(X[1] * H_Q, N) ** 2) * M, N)[::D_ADC]
    noise_std = np.sqrt(np.var(z0) * 10 ** (-SNR_ADC_DB / 10))
    _, spec_off = one_record(rng, 0, None, 0.0, noise_std)

    # ============ Part A：全带 burst 测 skew（I、Q 交替）
    print(f"\n[A] 全带 burst，ε = {EPS_BURST}，每支路 {T_BURST_MS} ms")
    B, sig, spec_on = run([(0, 0), (1, 0)], [None], EPS_BURST, noise_std, T_BURST_MS, procs, 100)
    skew_A = -np.angle(B[1] * np.conj(B[0])) / (2 * np.pi * nu)
    skew_A_sig = np.sqrt(sig[0] ** 2 + sig[1] ** 2) / (2 * np.pi * nu)
    # 全带 burst 的理论值：两路各自按权重 S|H(f)||H(f+ν)| 平均的群时延之差
    def line_tau(H, mask):
        Hs = np.interp(f + nu, f, H.real, right=0) + 1j * np.interp(f + nu, f, H.imag, right=0)
        return -np.angle(np.sum(RRC ** 2 * mask * Hs * np.conj(H))) / (2 * np.pi * nu)
    full = np.ones_like(f)
    skew_A_true = line_tau(H_Q, full) - line_tau(H_I, full)
    print(f"   skew 估计 {skew_A * 1e12:+.2f} ± {skew_A_sig * 1e12:.2f} ps   "
          f"（谱线理论值 {skew_A_true * 1e12:+.2f} ps，设定 {SKEW_PS} ps）")

    # ============ Part B：分片扫描测相频（每片 I、Q 各测一次）
    print(f"\n[B] {K} 片 × {Delta / 1e9:.0f} GHz 扫描，ε_s = {EPS_SLICE}，每片每支路 {T_SLICE_MS} ms")
    masks = [((f >= edges[k]) & (f < edges[k + 1])).astype(float) for k in range(K)]
    schedule = [(b, k) for k in range(K) for b in (0, 1)]
    B, sig, _ = run(schedule, masks, EPS_SLICE, noise_std, T_SLICE_MS, procs, 900)
    B = B.reshape(K, 2)
    sig = sig.reshape(K, 2)
    tau = -np.angle(B) / (2 * np.pi * nu)                 # 每片每支路的群时延（含 M 和 NCO 的共同常数）
    sig_tau = sig / (2 * np.pi * nu)
    tau_true = np.array([[line_tau(H, masks[k]) for H in (H_I, H_Q)] for k in range(K)])
    # 绝对延迟只能测到模 1/ν，且含 M(ν) 和 NCO 的共同常数：共同延迟本来就不可观测。
    # 测量值和真值各自减掉自己 I 路第 0 片的群时延（同一个线性参考项），只比较形状。
    ref = tau[0, 0]
    ref_true = tau_true[0, 0]
    tau = tau - ref
    tau_true = tau_true - ref_true
    # 逐片累加 → 相频（望远镜恒等式）
    fc = edges[:-1] + Delta / 2
    th = -2 * np.pi * np.concatenate((np.zeros((1, 2)), np.cumsum(tau * Delta, axis=0)))
    sig_th = 2 * np.pi * Delta * np.sqrt(np.concatenate((np.zeros((1, 2)), np.cumsum(sig_tau ** 2, axis=0))))
    idx = np.round(edges / f[1]).astype(int)
    th_true = np.stack([np.unwrap(np.angle(H))[idx] + 2 * np.pi * edges * ref_true for H in (H_I, H_Q)], axis=1)
    th_true -= th_true[0]
    d_tau = tau[:, 1] - tau[:, 0]
    sig_d = np.sqrt(sig_tau[:, 0] ** 2 + sig_tau[:, 1] ** 2)
    w = 1 / sig_d ** 2
    skew_B = np.sum(w * d_tau) / np.sum(w)
    skew_B_sig = 1 / np.sqrt(np.sum(w))
    print("   片中心(GHz)  τ_I 测/真 (ps)      τ_Q 测/真 (ps)      Δτ (ps)")
    for k in range(K):
        print(f"   {fc[k] / 1e9:6.0f}   {tau[k, 0] * 1e12:7.2f} / {tau_true[k, 0] * 1e12:6.2f}"
              f"   {tau[k, 1] * 1e12:7.2f} / {tau_true[k, 1] * 1e12:6.2f}"
              f"   {d_tau[k] * 1e12:+.2f} ± {sig_d[k] * 1e12:.2f}")
    print(f"   相频在 50 GHz 处：I 测 {th[-1, 0]:+.3f} / 真 {th_true[-1, 0]:+.3f} rad，"
          f"Q 测 {th[-1, 1]:+.3f} / 真 {th_true[-1, 1]:+.3f} rad（σ ≈ {sig_th[-1, 0]:.3f}）")
    print(f"   skew（逐片差分加权平均）{skew_B * 1e12:+.2f} ± {skew_B_sig * 1e12:.2f} ps（设定 {SKEW_PS} ps）")

    # ============ 画图
    import matplotlib
    if "--show" not in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fa = np.fft.rfftfreq(Nd, 1 / fs_adc) / 1e6
    ax[0, 0].semilogy(fa, spec_on / spec_on.max(), "r-", label=f"I-branch dither ON, eps={EPS_BURST} (avg of all records)")
    ax[0, 0].semilogy(fa, spec_off / spec_on.max(), "k--", alpha=0.6, label="dither OFF (1 record)")
    ax[0, 0].axvline(nu / 1e6, color="b", ls=":", label="nu = 90 MHz")
    ax[0, 0].set_xlabel("ADC frequency (MHz)"); ax[0, 0].set_ylabel("PSD (norm.)")
    ax[0, 0].set_title("What the 200 MS/s ADC sees: one coherent line on a data-beat floor")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    fcg = fc / 1e9
    for b, (c, mk, nm) in enumerate((("r", "o", "I"), ("g", "^", "Q"))):
        ax[0, 1].plot(fcg, tau_true[:, b] * 1e12, "-", color="k" if b == 0 else "b", label=f"true tau_g,{nm}")
        ax[0, 1].errorbar(fcg, tau[:, b] * 1e12, yerr=sig_tau[:, b] * 1e12, fmt=mk, color=c, capsize=3,
                          label=f"measured tau_g,{nm} = -arg(B)/(2 pi nu)")
    ax[0, 1].set_xlabel("slice centre (GHz)"); ax[0, 1].set_ylabel("group delay (ps), bulk removed")
    ax[0, 1].set_title("Part B: lock-in phase of each slice -> group delay"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    eg = edges / 1e9
    ax[1, 0].plot(eg, th_true[:, 0], "k-", lw=2, label="true theta_I")
    ax[1, 0].errorbar(eg, th[:, 0], yerr=sig_th[:, 0], fmt="ro", capsize=3, label="accumulated slice by slice")
    ax[1, 0].plot(eg, th_true[:, 1], "b-", lw=2, label="true theta_Q")
    ax[1, 0].errorbar(eg, th[:, 1], yerr=sig_th[:, 1], fmt="g^", capsize=3, label="accumulated")
    ax[1, 0].set_xlabel("frequency (GHz)"); ax[1, 0].set_ylabel("phase (rad), bulk delay removed")
    ax[1, 0].set_title("Phase response -6 (f/50 GHz)^2 rad, rebuilt from 5 slice group delays")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].errorbar(fcg, d_tau * 1e12, yerr=sig_d * 1e12, fmt="ro", capsize=3, label="per slice tau_Q - tau_I")
    ax[1, 1].axhline(SKEW_PS, color="k", ls="-", label=f"set skew {SKEW_PS} ps")
    ax[1, 1].axhline(skew_B * 1e12, color="r", ls="--", label=f"sweep mean {skew_B * 1e12:.2f} +- {skew_B_sig * 1e12:.2f} ps")
    ax[1, 1].axhline(skew_A * 1e12, color="b", ls=":", label=f"full-band burst {skew_A * 1e12:.2f} +- {skew_A_sig * 1e12:.2f} ps")
    ax[1, 1].set_xlabel("slice centre (GHz)"); ax[1, 1].set_ylabel("skew (ps)")
    ax[1, 1].set_title("I/Q skew = -arg(B_Q conj(B_I)) / (2 pi nu)"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
    fig.suptitle("Multiplicative pilot at 90 MHz, PD 100 MHz, ADC 200 MS/s, live 16QAM 100 GBaud, fresh data every record",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("standalone_pilot_demo.png", dpi=130)
    print("\n图已保存: standalone_pilot_demo.png")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    freeze_support()
    main()
