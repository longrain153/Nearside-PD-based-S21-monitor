"""Build docs/dither_principle_derivation.pdf (Chinese, embedded WQY font,
formulas rendered by matplotlib mathtext)."""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, KeepTogether)

REPO = "/home/user/Nearside-PD-based-S21-monitor"
FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
OUT = os.path.join(REPO, "docs", "dither_principle_derivation.pdf")
FIGDIR = os.path.join(REPO, "docs", "figures")
pdfmetrics.registerFont(TTFont("WQY", FONT))
f_cn = font_manager.FontProperties(fname=FONT)
plt.rcParams["mathtext.fontset"] = "cm"

W, H = A4
body = ParagraphStyle("body", fontName="WQY", fontSize=10.5, leading=17,
                      alignment=TA_LEFT, wordWrap="CJK", spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=14, textColor=colors.HexColor("#444444"))
h1 = ParagraphStyle("h1", fontName="WQY", fontSize=15, leading=22, spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1a3d6d"))
h2 = ParagraphStyle("h2", fontName="WQY", fontSize=12, leading=18, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1a3d6d"))
title_st = ParagraphStyle("title", fontName="WQY", fontSize=20, leading=28, alignment=TA_CENTER, spaceAfter=8)
sub_st = ParagraphStyle("sub", fontName="WQY", fontSize=11, leading=16, alignment=TA_CENTER,
                        textColor=colors.HexColor("#555555"), spaceAfter=4)
cap = ParagraphStyle("cap", fontName="WQY", fontSize=9, leading=13, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=8)


def P(t, st=body):
    t = t.replace("<b>", '<font color="#1a3d6d">').replace("</b>", "</font>")
    return Paragraph(t, st)


def eq(tex, size=12, label=None):
    """Render a display formula with mathtext to an Image flowable."""
    fig = plt.figure(figsize=(0.1, 0.1))
    txt = fig.text(0, 0, f"${tex}$", fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    im = PILImage.open(buf)
    w_pt = im.width / 300 * 72
    h_pt = im.height / 300 * 72
    maxw = W - 50 * mm
    if w_pt > maxw:
        h_pt *= maxw / w_pt
        w_pt = maxw
    buf.seek(0)
    img = Image(buf, width=w_pt, height=h_pt)
    if label:
        t = Table([[img, P(label, small)]], colWidths=[W - 60 * mm, 16 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("ALIGN", (0, 0), (0, 0), "CENTER"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t
    img.hAlign = "CENTER"
    return img


def fig_image(path, width_mm):
    from PIL import Image as PILImage
    im = PILImage.open(path)
    w = width_mm * mm
    h = w * im.height / im.width
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    return img


def table(rows, colw, header=True, fs=9):
    data = [[P(c, ParagraphStyle("c", fontName="WQY", fontSize=fs, leading=fs + 4, wordWrap="CJK")) for c in r] for r in rows]
    t = Table(data, colWidths=colw, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7"))]
    t.setStyle(TableStyle(st))
    return t


# ----------------------------------------------------------------------------
# illustrative figures generated here
# ----------------------------------------------------------------------------
def make_block_diagram():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off")
    boxes = [(2, 14, 16, 8, "发端 DSP\n数据 + 乘性导频\n(720 点 FFT)"),
             (22, 18, 14, 6, "DAC/驱动/调制器 I\nH_I(f)"),
             (22, 8, 14, 6, "DAC/驱动/调制器 Q\nH_Q(f)"),
             (40, 13, 10, 6, "光场\ny_I + j y_Q"),
             (54, 13, 12, 6, "近端 PD\n|·|² 平方律"),
             (70, 13, 12, 6, "前端 M(f)\n≈100 MHz"),
             (86, 13, 12, 6, "ADC 200 MS/s\n锁相取 ν")]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#eef3fa", ec="#1a3d6d", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5, fontproperties=f_cn)
    arr = dict(arrowstyle="->", color="#333333", lw=1.1)
    ax.annotate("", (22, 21), (18, 18.5), arrowprops=arr)
    ax.annotate("", (22, 11), (18, 17.5), arrowprops=arr)
    ax.annotate("", (40, 16.5), (36, 21), arrowprops=arr)
    ax.annotate("", (40, 15.5), (36, 11), arrowprops=arr)
    for x0, x1 in ((50, 54), (66, 70), (82, 86)):
        ax.annotate("", (x1, 16), (x0, 16), arrowprops=arr)
    ax.text(10, 5, "监测端不知道数据、不知道 M(f)、不与发端同步相位", fontsize=8.5, color="#666666", fontproperties=f_cn)
    p = os.path.join(FIGDIR, "principle_block_diagram.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def make_spectrum_illustration():
    f = np.linspace(-70, 70, 2000)
    def rrc(f, B=50, r=0.1):
        a = np.ones_like(f); x = np.abs(f)
        m = (x > B * (1 - r)) & (x < B * (1 + r))
        a[m] = 0.5 * (1 + np.cos(np.pi / (2 * r * B) * (x[m] - B * (1 - r))))
        a[x >= B * (1 + r)] = 0
        return a
    nu = 6.0  # exaggerated for the sketch
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), gridspec_kw=dict(hspace=0.55))
    ax[0].plot(f, rrc(f), color="#2a78d6", lw=2, label="数据谱 S(f)（载体）")
    ax[0].plot(f, 0.35 * rrc(f - nu), color="#eb6834", lw=1.6, ls="--", label="副本 (ε/2)·X(f−ν)")
    ax[0].plot(f, 0.35 * rrc(f + nu), color="#eb6834", lw=1.6, ls=":", label="副本 (ε/2)·X(f+ν)")
    for f0 in (-40, -20, 10, 30, 45):
        ax[0].annotate("", (f0 + nu, 0.35), (f0, 1.0), arrowprops=dict(arrowstyle="<->", color="#1baf7a", lw=1.3))
    ax[0].text(48, 0.62, "每个 f 都有一对 (f, f+ν)\n同源相干，相位差 = θ(f+ν)−θ(f)", fontsize=9,
               fontproperties=f_cn, color="#1baf7a", ha="left")
    ax[0].set_xlim(-70, 75); ax[0].set_ylim(0, 1.25)
    ax[0].set_xlabel("光场基带频率 (GHz)，ν 为示意放大", fontproperties=f_cn, fontsize=9)
    ax[0].set_title("发端：乘性导频 = 整个数据谱平移 ±ν 的两份副本", fontproperties=f_cn, fontsize=10)
    ax[0].legend(prop=f_cn, fontsize=8, loc="upper left", frameon=False); ax[0].set_yticks([])

    fp = np.linspace(0, 100, 2000)
    floor = np.exp(-(fp / 60) ** 2) * 0.6 + 0.02
    ax[1].fill_between(fp, 0, floor, color="#c9d6ea", label="数据自拍 y_d²（宽带、零均值，主要噪声）")
    ax[1].plot([nu * 3, nu * 3], [0, 1.1], color="#eb6834", lw=2.5, label="相干谱线 2·y_d·dy 在 ν 处（信号）")
    ax[1].axvspan(0, 24, color="#1baf7a", alpha=0.08)
    ax[1].text(12, 0.95, "PD/ADC 带宽\n（≈100 MHz）", ha="center", fontsize=9, fontproperties=f_cn, color="#1baf7a")
    ax[1].set_xlim(0, 100); ax[1].set_ylim(0, 1.2); ax[1].set_yticks([])
    ax[1].set_xlabel("PD 输出频率（示意）", fontproperties=f_cn, fontsize=9)
    ax[1].set_title("PD 平方律输出：所有 (f, f+ν) 对都拍到同一个 ν 上，相干叠加成一条线", fontproperties=f_cn, fontsize=10)
    ax[1].legend(prop=f_cn, fontsize=8, loc="upper right", frameon=False)
    p = os.path.join(FIGDIR, "principle_spectrum_sketch.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


blk = make_block_diagram()
sk = make_spectrum_illustration()

# ----------------------------------------------------------------------------
story = []
S = story.append

S(Spacer(1, 30 * mm))
S(P("基于乘性导频的相干发射机在线 S21 监测", title_st))
S(P("原理推导：为什么一个 100 MHz 量级的近端光电探测器能测出 50 GHz 带宽的幅频、相频和 I/Q skew", sub_st))
S(Spacer(1, 6 * mm))
S(P("版本 2026-09-30 · 配套代码 examples/phase_sweep_fft720.py、examples/dither_demo_90mhz.py", sub_st))
S(Spacer(1, 14 * mm))
S(P("<b>摘要</b>：在发端 DSP 里给某一支路（或某一段频谱）的数据乘一个慢变增益 1+ε·cos(2πνt)，"
    "ν 为几十到几百 MHz。数据谱因此被复制两份并平移 ±ν，每一个频率 f 上都出现一对同源相干的分量 (f, f+ν)。"
    "它们经过支路响应 H(f) 后在平方律 PD 里互拍，所有对都落在同一个频率 ν 上并相干叠加成一条谱线，"
    "谱线相位等于 θ(f+ν)−θ(f) 的加权平均，即支路群时延。数据自身的随机相位在互拍中精确抵消，"
    "PD 前端响应 M(ν) 和导频初相对 I、Q 两路、对所有频段完全相同，在差分和扫频中被消去。"
    "本文从平方律展开出发，逐步推导谱线的均值、噪声、精度公式，说明 skew、相频、幅频各自怎么提取，"
    "给出系统性误差来源和工程约束，并用逐样本仿真验证。", body))
S(PageBreak())

# ---------------------------------------------------------------- 1
S(P("1. 问题设定", h1))
S(P("相干发射机的 I、Q 两个驱动支路各有一个从 DSP 输出到光场的线性响应，记为 H<sub>I</sub>(f) 和 H<sub>Q</sub>(f)。"
    "在本文关心的场景里正交误差为零，两条支路可以分开处理：", body))
S(eq(r"y_I(t) = (h_I * x_I)(t),\qquad y_Q(t) = (h_Q * x_Q)(t),\qquad H_X(f)=|H_X(f)|\,e^{j\theta_X(f)}", 12, "(1.1)"))
S(P("要在线测的是三个量：幅频 |H<sub>X</sub>(f)|、相频 θ<sub>X</sub>(f)（去掉不可观测的常数和线性项）、"
    "以及 I/Q skew，即两条支路群时延之差。群时延的定义是", body))
S(eq(r"\tau_{g,X}(f) = -\frac{1}{2\pi}\,\frac{d\theta_X}{df},\qquad \mathrm{skew} = \tau_{g,Q}-\tau_{g,I}", 12, "(1.2)"))
S(P("监测端只有一个近端光电探测器（PD），输出电流正比于光强 |y<sub>I</sub>+jy<sub>Q</sub>|² = y<sub>I</sub>²+y<sub>Q</sub>²，"
    "后接一个带宽约 100 MHz 的前端 M(f) 和一个 200 MS/s 的 ADC。监测端拿不到 DSP 数据，"
    "不与发端时钟同相位，也不精确知道 M(f)。图 1 是整条链路。", body))
S(fig_image(blk, 165))
S(P("图 1：链路框图。所有待测量都在 50 GHz 带宽的 H<sub>I</sub>、H<sub>Q</sub> 里，读出只有 100 MHz。", cap))
S(P("难点在于 PD 是平方律器件：它丢掉光场的绝对相位，而且它的带宽比信号窄 500 倍。"
    "本方案不试图让 PD “看见”高频，而是让发端把高频段的相位信息主动搬到 PD 带内的一个频点上。", body))

# ---------------------------------------------------------------- 2
S(P("2. 乘性导频：定义与频域含义", h1))
S(P("2.1 时域定义", h2))
S(P("对支路 X 的 DSP 输出乘一个慢变增益：", body))
S(eq(r"x_X(t)\;\rightarrow\;x_X(t)\,\left[1+\varepsilon\cos(2\pi\nu t+\varphi)\right] = x_X(t)+d_X(t)", 12, "(2.1)"))
S(P("ε 是调制深度（典型 0.08–0.3），ν 是导频频率（典型 76–90 MHz，须落在 PD 和 ADC 带内），φ 是初相。"
    "“乘性”与“加性”的区别是根本性的：不是往信号里加一根正弦音，而是把数据本身当载波去调幅。"
    "d<sub>X</sub> 称为导频扰动。", body))
S(P("2.2 频域含义：整谱平移", h2))
S(P("时域相乘等于频域卷积，cos 的谱只有 ±ν 两根线，因此", body))
S(eq(r"D_X(f) = \frac{\varepsilon}{2}\,e^{j\varphi}X_X(f-\nu)+\frac{\varepsilon}{2}\,e^{-j\varphi}X_X(f+\nu)", 12, "(2.2)"))
S(P("整个数据谱被复制两份，分别平移 +ν 和 −ν，幅度 ε/2。导频扰动因此不是一根谱线，而是一个和数据一样宽的“影子谱”。"
    "关键结论是：对带内每一个频率 f，f+ν 处都出现了 X(f) 的一个副本；副本和原分量携带同一个数据符号，是同源相干的。"
    "图 2 上半部分示意了这一点。", body))
S(fig_image(sk, 160))
S(P("图 2：上：乘性导频把数据谱平移 ±ν，每个 f 都配成一对；下：PD 输出中所有对都拍到 ν 上叠成一条线，"
    "数据自拍是宽带零均值的噪声底。ν 在图中放大了约 60 倍以便看清。", cap))
S(P("2.3 在 720 点块 FFT 里的实现", h2))
S(P("发端频域成型用 720 点 FFT 分块处理，块长 720 个样本（200 GS/s 下 3.6 ns），bin 间距 Δf<sub>bin</sub> = 277.8 MHz。"
    "若 ν = m·Δf<sub>bin</sub> 是 bin 间距的整数倍，则因为 e<sup>j2πmn/720</sup> 恰好是 DFT 的基函数，(2.1) 在频域上精确等于常系数的邻 bin 混合：", body))
S(eq(r"X_k \;\rightarrow\; X_k+\frac{\varepsilon}{2}e^{j\varphi}X_{k-m}+\frac{\varepsilon}{2}e^{-j\varphi}X_{k+m}", 12, "(2.3)"))
S(P("代码里不出现 cos，cos 的时变已经被下标平移吃掉。若 ν 低于 bin 间距（如 90 MHz），"
    "则改为块级标量增益：第 b 块片内所有 bin 乘同一个 g<sub>b</sub> = 1+ε·cos(2πν·t<sub>b</sub>)，t<sub>b</sub> 为该块的中心时刻。"
    "这相当于把 cos 用块速率采样后零阶保持，基波幅度乘 sinc(ν/Δf<sub>bin</sub>)（90 MHz 时为 0.836），"
    "镜像落在 Δf<sub>bin</sub>±ν 处，在 PD 带外。两种做法都不需要新的硬件模块。", body))
S(P("2.4 对主数据流的代价", h2))
S(P("导频扰动的功率是被调制部分功率的 ε²/2。若只调制带宽 B<sub>d</sub> 的一片，相对整个数据功率的占比为", body))
S(eq(r"p=\frac{\varepsilon^2}{2}\cdot\frac{P_{\mathrm{slice}}}{P_{\mathrm{total}}}\approx\frac{\varepsilon^2}{2}\cdot\frac{B_d}{B_{\mathrm{sig}}}", 12, "(2.4)"))
S(P("全带 ε = 0.08 对应 p = 0.32%（−25 dB），对 EVM 无感；这是本文的功率预算基准。", body))

# ---------------------------------------------------------------- 3
S(P("3. 平方律 PD 的输出：A、B、C 三项", h1))
S(P("支路输出写成数据部分加导频部分，y<sub>X</sub> = y<sub>d,X</sub> + dy<sub>X</sub>，其中 dy<sub>X</sub> = h<sub>X</sub> * d<sub>X</sub>。"
    "只在 I 路加导频时，光强为", body))
S(eq(r"p(t)=(y_{d,I}+dy_I)^2+y_{d,Q}^2=\;(y_{d,I}^2+y_{d,Q}^2)\;+\;2\,y_{d,I}\,dy_I\;+\;dy_I^2", 12, "(3.1)"))
S(P("右边三项依次记为 A、B、C：", body))
S(table([["项", "阶数", "在 ν 处的均值", "作用"],
         ["A = y<sub>d</sub>²", "ε⁰", "零。数据平稳，其功率的期望是常数，ν 处只有随机起伏", "主要噪声：数据自拍的宽带底"],
         ["B = 2·y<sub>d</sub>·dy", "ε¹", "非零。含 |X(f)|² 项，见第 4 节", "信号：相干谱线"],
         ["C = dy²", "ε²", "零。dy 是影子谱经 H，其自拍是宽带零均值的", "可忽略：功率 ∝ ε⁴"]],
        [30 * mm, 14 * mm, 70 * mm, 46 * mm]))
S(Spacer(1, 4))
S(P("A 项虽然均值为零，但它是整个数据带宽自拍下来的宽带噪声，在 ν 附近有一个平坦的功率密度，"
    "这是本方案精度的根本限制（第 7 节）。C 项要注意它不是“直流 + 2ν”：对乘性调制，dy 是宽带信号，"
    "dy² 在 ν 处是零均值的随机量，只贡献 ∝ ε⁴ 的噪声。", body))

# ---------------------------------------------------------------- 4
S(P("4. 谱线的均值：为什么 PD 能看到相位", h1))
S(P("4.1 相干双音的拍频", h2))
S(P("先看两个音。支路输出中若同时存在", body))
S(eq(r"a\cos(2\pi f t+\alpha)+b\cos\left(2\pi(f+\nu)t+\beta\right)", 12))
S(P("平方后除高频项外出现 a·b·cos(2πνt + β − α)。PD 丢掉了绝对相位 α，但两音的相位差 β−α 完整保留在 ν 处。"
    "这是所有外差测量的基础。", body))
S(P("4.2 数据相位的精确抵消", h2))
S(P("没有导频时，f 和 f+ν 处是两个独立符号 X(f)、X(f+ν)，相位差随机，拍频平均为零，这就是 A 项。"
    "加了乘性导频后，f+ν 处多出 X(f) 的副本，两音写出来是", body))
S(eq(r"f:\;X(f)H(f),\qquad f+\nu:\;\frac{\varepsilon}{2}e^{j\varphi}X(f)H(f+\nu)", 12, "(4.1)"))
S(P("同一个 X(f)，数据自身的随机相位在相减时精确消去：", body))
S(eq(r"\beta-\alpha=\theta(f+\nu)-\theta(f)+\varphi", 12, "(4.2)"))
S(P("剩下的只有支路响应 H 的相位差。它对每个 f 都是确定值，不随符号变，所以全带叠加起来是相干累加而不是抵消。", body))
S(P("4.3 B 项的期望", h2))
S(P("把 (2.2) 代入 B = 2y<sub>d</sub>·dy，对数据取期望，用 E[X(f)X*(f')] = S(f)δ(f−f')（S 为数据功率谱），"
    "得到 PD 输出中 ν 处复振幅的均值（取正频分量）：", body))
S(eq(r"\overline{B}(\nu)=\varepsilon\,e^{j\varphi}\int S(f)\,H(f+\nu)\,H^*(f)\,df", 13, "(4.3)"))
S(P("−ν 那份副本给出 ∫S(f)H(f)H*(f−ν)df，换元 f→f+ν 后与上式同形，只是权重换成 S(f+ν)，"
    "两者合并相当于权重取 [S(f)+S(f+ν)]/2，对结论没有影响。经过 PD 前端后，ADC 上看到的谱线是", body))
S(eq(r"\tilde B_X(\nu)=M(\nu)\,\overline{B}_X(\nu)=\varepsilon\,e^{j\varphi}M(\nu)\int S(f)\,|H_X(f)||H_X(f+\nu)|\,e^{j[\theta_X(f+\nu)-\theta_X(f)]}\,df", 11.5, "(4.4)"))
S(P("4.4 相位差就是群时延", h2))
S(P("ν（90 MHz）远小于 H 变化的尺度（GHz），相位差是局部导数：", body))
S(eq(r"\theta(f+\nu)-\theta(f)\approx\nu\,\frac{d\theta}{df}=-2\pi\nu\,\tau_g(f)", 12, "(4.5)"))
S(P("代入 (4.4)，谱线相位为", body))
S(eq(r"\arg\tilde B_X=-2\pi\nu\,\langle\tau_{g,X}\rangle_w+\arg M(\nu)+\varphi,\qquad w(f)\propto S(f)|H_X(f)||H_X(f+\nu)|", 12, "(4.6)"))
S(P("〈·〉<sub>w</sub> 是按权重 w 对被调制频段取的平均。这就是核心结果：ADC 在 ν 处读到的一根谱线，"
    "其相位携带的是数据频段上支路群时延的加权平均，而不是 ν 那个频点自己的响应。"
    "带宽再宽，PD 看到的只是 ν 频率的强度包络，包络经支路延迟了 τ<sub>g</sub>，在 ν 上就是 −2πντ<sub>g</sub> 的相位。"
    "这正是测群时延的经典“调制相移法”，只是载波换成了数据本身。", body))
S(P("4.5 谱线的幅度", h2))
S(P("由 (4.4)，|B̃<sub>X</sub>| ≈ ε|M(ν)|∫S|H<sub>X</sub>|²df（ν 远小于 |H| 的变化尺度时 |H(f+ν)|≈|H(f)|）。"
    "只对第 k 片频谱加导频时，积分只在片内进行，于是 |B̃<sub>X</sub>(k)| 给出片内 |H<sub>X</sub>|² 的加权积分，"
    "这是幅频测量的依据（第 6.3 节）。", body))

# ---------------------------------------------------------------- 5
S(P("5. 谱线的提取：数字锁相", h1))
S(P("ADC 记录 z[n]（去均值）与本地复指数相关：", body))
S(eq(r"\tilde B=\sum_n w[n]\,z[n]\,e^{-j2\pi\nu\,(t_0+n/f_{\mathrm{adc}})}", 12, "(5.1)"))
S(P("w[n] 是 Hann 窗，用来压掉换片过渡和数据自拍在 ν 附近的泄漏；t<sub>0</sub> 是这段记录的全局起始时刻。"
    "ν 不必落在 DFT 网格上。z 中 ν 处的成分是 A·cos(2πνt+ψ)，乘 e<sup>−j2πνt</sup> 后正频分量搬到直流并被求和累积，"
    "负频分量搬到 −2ν 被求和压掉，宽带噪声累加后按 1/√N 收敛。于是 arg B̃ = ψ + 噪声。", body))
S(P("<b>长积分的累加方式</b>：不要每块各算一个角度再平均角度，而是所有块的复数 B̃ 直接相加，最后取一次辐角。"
    "这样噪声按 1/√T 收敛，前提是块与块之间相位可加。", body))
S(P("<b>初相的处理</b>：发端的 ν 发生器（NCO）上电后连续跑、永不复位，换支路、换片只改掩码不碰 NCO；"
    "收端用同一个全局时间做 (5.1)。这样 φ、arg M(ν) 和 ADC 时钟相位差对所有测量是同一个常数，"
    "在差分（skew）和扫频（相频，减参考项）中被消去。剩下的要求只有发端与收端时钟的频率差 δν："
    "最稳妥是 ADC 时钟与发端 DSP 同源，此时 δν = 0；否则可以从数据里联合估计 ν̂，或常驻一个参考片"
    "（频率 ν+δ，δ 约 1 kHz）让时钟漂移在两根线之间抵消。", body))

# ---------------------------------------------------------------- 6
S(P("6. 三个量各自怎么提取", h1))
S(P("6.1 skew：全带 burst，I/Q 相减", h2))
S(P("时段 A 对 I 路全带加导频，时段 B 对 Q 路，两段用同一个 NCO。由 (4.6)：", body))
S(eq(r"\arg\tilde B_Q-\arg\tilde B_I=-2\pi\nu\left(\langle\tau_{g,Q}\rangle-\langle\tau_{g,I}\rangle\right)", 12, "(6.1)"))
S(eq(r"\widehat{\mathrm{skew}}=-\frac{\arg\left(\tilde B_Q\,\tilde B_I^{*}\right)}{2\pi\nu}", 13, "(6.2)"))
S(P("arg M(ν)、φ 和 ADC 时钟相位在同一个 ν 下精确抵消；不需要知道 M、光功率、|H| 或 DSP 数据。"
    "先相乘再取角，避免解卷绕；相位差是毫弧度量级，不会越过 ±π。要测的相位很小："
    "ν = 90 MHz 时 3 ps 对应 1.7 mrad，0.1 ps 对应 57 µrad。", body))
S(P("<b>幅频权重偏置</b>。(6.1) 中两个 ⟨τ<sub>g</sub>⟩ 的权重 w<sub>I</sub>、w<sub>Q</sub> 不同（|H<sub>I</sub>|≠|H<sub>Q</sub>|）。"
    "若两路还有共同的、随频率变化的群时延 τ<sub>c</sub>(f)，则", body))
S(eq(r"\langle\tau_{g,Q}\rangle-\langle\tau_{g,I}\rangle=\mathrm{skew}+\int\left[w_Q(f)-w_I(f)\right]\,\tau_c(f)\,df", 12, "(6.3)"))
S(P("第二项是系统偏置。本文场景（公共二次相位 6 rad，|H<sub>Q</sub>| 滚降更早）中权重重心相差 1.2 GHz，"
    "τ<sub>c</sub> 斜率 0.76 ps/GHz，偏置 0.94 ps，远大于 0.1 ps 目标。三种消除办法：（a）逐片差分再平均，"
    "片内 I/Q 重心几乎重合（内部各片偏置 &lt;0.004 ps，只有滚降片例外，去掉即可）；"
    "（b）burst 只在两路幅频平坦一致的中频段做，如 10–40 GHz，信噪比只损失 √(30/55)；"
    "（c）用扫频测得的 |H| 和 τ<sub>g</sub>(f) 算出 (6.3) 第二项并扣除。", body))

S(P("6.2 相频：连续分片扫描与望远镜累加", h2))
S(P("一个 ν 覆盖全带只能给一个数（平均群时延）。要得到曲线，把导频限制在第 k 片 [f<sub>k</sub>, f<sub>k</sub>+Δ] 内"
    "（720 点 FFT 下即 18 个 bin = 5 GHz），ν 不变，逐片测量。片内每个 f 都有一对 (f, f+ν)，谱线相位是片内平均斜率。"
    "对这个平均做积分换元：", body))
S(eq(r"\frac{1}{\Delta}\int_{f_k}^{f_k+\Delta}\left[\theta(f+\nu)-\theta(f)\right]df=\frac{1}{\Delta}\left[\int_{f_k+\Delta}^{f_k+\Delta+\nu}\theta\,df-\int_{f_k}^{f_k+\nu}\theta\,df\right]=\frac{\nu}{\Delta}\left[\bar\theta(f_k+\Delta)-\bar\theta(f_k)\right]", 11, "(6.4)"))
S(P("θ̄ 是 θ 在 ν 宽窗内的平均。也就是说，一片给出的是 θ̄ 在片两端的差，中间 f 到 f+ν 的细节不需要知道；"
    "这是微积分基本定理：区间内导数的平均等于两端函数值之差除以区间长。于是", body))
S(eq(r"\tau_{g}(k)=\frac{\arg M(\nu)-\arg\tilde B(k)}{2\pi\nu},\qquad \bar\theta(f_{k+1})=\bar\theta(f_k)-2\pi\Delta\,\tau_g(k)", 12, "(6.5)"))
S(P("片首尾相接铺满整个带时，求和是望远镜式的：θ̄(f<sub>1</sub>)−θ̄(f<sub>0</sub>) 加 θ̄(f<sub>2</sub>)−θ̄(f<sub>1</sub>) …，"
    "中间项全部抵消，每个节点都直接得到，不依赖片之间怎么变化的任何假设。不可观测的只有两样：常数项（φ 和 arg M）"
    "和线性项（共同延迟），两者都不是 S21 的可观测量。分辨率有三层：曲线点间距由片宽 Δ 决定；"
    "能看到的最细频率结构由 ν 决定（θ 被 ν 窗平滑）；覆盖范围由数据带宽决定。"
    "对 ν 的进一步说明：ν 只是“尺子的长度”，不是量程，尺子可以放到带内任何位置。", body))
S(P("<b>关于 (6.4) 的权重</b>：(4.6) 的平均带权重 w ∝ S|H(f)||H(f+ν)|，(6.4) 假设片内权重平坦。"
    "带边滚降片会把平均向幅度大的一侧偏，幅频本来要测，可用测得的 |H| 反解，是二阶小量。", body))
S(P("<b>模型拟合</b>：发射机相频通常光滑，可对 τ<sub>g</sub>(k) 做低阶多项式拟合（二次相位对应线性群时延，2 个参数），"
    "比逐点积分精度高数倍。", body))

S(P("6.3 幅频：谱线幅度", h2))
S(eq(r"|\tilde B_X(k)|\approx\varepsilon_s\,|M(\nu)|\int_{\mathrm{slice}\,k}S(f)|H_X(f)|^2df\quad\Rightarrow\quad|H_X(f_k)|\propto\sqrt{\frac{|\tilde B_X(k)|}{\varepsilon_s|M(\nu)|\,P_S(k)}}", 12, "(6.6)"))
S(P("P<sub>S</sub>(k) 是片内数据功率，由成型滤波器已知。|M(ν)| 对所有片相同，归一化后消去，得到相对幅频。"
    "报告的第一版扫频仿真中 |H| 最大误差 0.1 dB 量级。", body))

# ---------------------------------------------------------------- 7
S(P("7. 噪声与精度", h1))
S(P("7.1 噪声来源", h2))
S(P("在锁相带宽 1/T 内，主要噪声是 A 项：全带数据自拍在 ν 附近的功率密度。它与光功率成正比，"
    "而谱线幅度也与光功率成正比，所以光功率在信噪比中抵消。ADC 热噪声比 A 项低 30 dB，可忽略。"
    "A 项在 ν 处的功率密度正比于 ∫S²|H|⁴df ∝ S²B<sub>sig</sub>（平坦谱近似），谱线幅度正比于 ε·S·B<sub>d</sub>，"
    "B<sub>d</sub> 为被调制的带宽。于是 T 秒积分后的幅度信噪比为", body))
S(eq(r"\rho\;\propto\;\varepsilon\,\frac{B_d}{\sqrt{B_{\mathrm{sig}}}}\sqrt{T}\qquad(\text{全带时 }B_d=B_{\mathrm{sig}}:\;\rho=\varepsilon\sqrt{B_{\mathrm{sig}}T})".replace(r"\text{全带时 }", r"\mathrm{full\ band}\ "), 12, "(7.1)"))
S(P("7.2 精度公式", h2))
S(P("谱线相位的标准差是 1/ρ。skew 由两条支路各积分 T 的两个相位相减得到，误差 √2 倍，除以 2πν：", body))
S(eq(r"\sigma_{\mathrm{skew}}=\frac{\sqrt{2}}{2\pi\nu\,\varepsilon\sqrt{B_{\mathrm{sig}}T}}\qquad(\text{全带})".replace(r"\text{全带}", r"\mathrm{full\ band}"), 13, "(7.2)"))
S(P("代入 ν = 90 MHz、ε = 0.08、B<sub>sig</sub> = 55 GHz、T = 5.24 ms 得 1.84 ps，与去相关仿真的 1.95±0.16 ps 一致。"
    "把 ε 换成功率占比 p = ε²B<sub>d</sub>/(2B<sub>sig</sub>)，公式变成更有用的形式：", body))
S(eq(r"\sigma_{\mathrm{skew}}=\frac{1}{2\pi\nu\sqrt{p\,B_d\,T}}\qquad\Rightarrow\qquad T\propto\frac{1}{\nu^2\,p\,B_d}", 13, "(7.3)"))
S(P("三个结论：（1）固定 EVM 预算 p 时，导频铺得越宽越好，因为谱线幅度 ∝ ε·P<sub>d</sub> = √p·√P<sub>d</sub>，"
    "把预算集中到窄片上反而亏 √(B<sub>sig</sub>/B<sub>d</sub>)。（2）时间与 ν² 成反比：要测的相位 2πν·skew 随 ν 放大，"
    "而相位噪声不随 ν 变。（3）光功率和 M 都不进公式。", body))
S(P("7.3 −25 dB 预算下达到 0.1 ps skew 的时间", h2))
S(table([["方式", "B<sub>d</sub>", "ε", "理论总时间", "含块级 ZOH 损失 ×1.43"],
         ["2 bin 片扫 100 片（ε<sub>s</sub>=0.8）", "0.56 GHz", "0.8", "350 s", "500 s"],
         ["5 GHz 片扫 10 片（本次仿真）", "5 GHz", "0.253", "40 s", "57 s"],
         ["中频段 burst 10–40 GHz", "30 GHz", "0.108", "6.6 s", "9.4 s"],
         ["全带 burst", "55 GHz", "0.08", "3.6 s", "5.2 s"],
         ["全带 burst + 发端预对消（方案 A）", "55 GHz", "0.08", "0.16 s", "0.23 s"]],
        [58 * mm, 22 * mm, 16 * mm, 28 * mm, 36 * mm]))
S(Spacer(1, 4))
S(P("发端预对消是指发端用自己的数据和 Ĥ 预测读出带内的数据自拍并注入共模增益抵消它，可把 A 项压低约 22 倍时间。"
    "ν 提高到 bin 间距 277.8 MHz 并改用邻 bin MAC 后，全带 burst 降到 0.38 s，图 3 给出随 ν 的变化。", body))
S(fig_image(os.path.join(FIGDIR, "nu_scaling.png"), 150))
S(P("图 3：所需总积分时间随导频频率的变化，T ∝ 1/ν²，dither 功率 −25 dB。", cap))
S(P("7.4 相频的精度", h2))
S(P("每片 τ<sub>g</sub> 的标准差按 (7.3) 用 B<sub>d</sub> = Δ、T = T<sub>片</sub> 计算，逐片累加的相位误差按随机游走增长：", body))
S(eq(r"\sigma_{\theta}(f_K)=2\pi\Delta\sqrt{\sum_{k<K}\sigma_{\tau}^2(k)}", 12, "(7.4)"))
S(P("5 GHz 片、−25 dB、每片每支路 8 ms 时每片约 3 ps，50 GHz 处约 0.3 rad；要 0.1 rad 需 10 倍时间。"
    "二次模型拟合只估曲率一个参数，同样数据下误差更小。", body))

# ---------------------------------------------------------------- 8
S(P("8. 系统性误差与工程要求", h1))
S(table([["项目", "机理", "要求 / 处理"],
         ["幅频权重偏置", "(6.3)：I/Q 权重重心不同 × 公共群时延斜率", "逐片差分、中频段 burst 或按测得 |H| 校正；本场景偏置 0.94 ps"],
         ["块级增益的时间戳", "块内样本 0…L−1 的中心是 (L−1)/2 而非 L/2，差半个样本 = 2.5 ps", "共同延迟，skew 和曲线形状不受影响；做绝对延迟时要对准保持区间真实中心"],
         ["ZOH 镜像", "块级增益的镜像在 Δf<sub>bin</sub>±ν；ν 副本与 (Δf<sub>bin</sub>−ν) 副本互拍出 |2ν−Δf<sub>bin</sub>| 线", "ν 避开 Δf<sub>bin</sub>/3（92.6 MHz）附近；仿真中 97.8 MHz 处可见此线"],
         ["时钟频差 δν", "相位按 2πδν·t 漂移", "ADC 与发端时钟同源；或联合估计 ν̂；或常驻参考片。相对漂移 &lt; 0.1 ps"],
         ["ν 稳定度", "T 内相位跑动", "约 5 ppm（1/T 级）"],
         ["换片过渡", "M 的群时延（20 ns）和振铃", "每片开头丢弃几 µs 保护间隔，Hann 窗"],
         ["数据自拍泄漏", "记录边界处的确定性瞬态", "Hann 窗，否则信噪比封顶在约 28×"],
         ["DFT 符号约定", "e<sup>−j</sup>/e<sup>+j</sup> 任一处反了 skew 变号", "按设计固定，用已知 skew 校验一次"],
         ["有限差分分辨率", "θ 被 ν 窗平滑", "看不到比 ν 更窄的结构；发射机 S21 在 GHz 尺度变化，无影响"]],
        [30 * mm, 66 * mm, 64 * mm], fs=8.5))

# ---------------------------------------------------------------- 9
S(P("9. 逐样本仿真验证（720 点 FFT 实现）", h1))
S(P("仿真按 200 GS/s 逐样本进行：发端 720 点块 FFT 中对片内 18 个 bin 乘块级增益，ν = 90 MHz；"
    "H<sub>I</sub>、H<sub>Q</sub> 带公共二次相位 −6(f/50 GHz)² rad，Q 路晚 0.3 ps，|H<sub>I</sub>|≠|H<sub>Q</sub>|；"
    "PD 平方律 + 100 MHz 前端 + 200 MS/s ADC + 30 dB 热噪声；每条 1.08 µs 记录都用新生成的 16QAM 数据，"
    "无任何对消。10 片 × 5 GHz 首尾相接，每片每支路 8 ms，总数据 160 ms。", body))
S(fig_image(os.path.join(FIGDIR, "phase_sweep_fft720_eps0p8.png"), 165))
S(P("图 4：ε<sub>s</sub> = 0.8 的仿真结果。左上：ADC 频谱中的 90 MHz 谱线与自拍噪声底；右上：每片群时延；"
    "左下：望远镜累加得到的相频与真值；右下：逐片 Δτ 与 skew 估计。", cap))
S(table([["", "ε<sub>s</sub> = 0.8（−14.9 dB）", "ε<sub>s</sub> = 0.253（−25 dB）"],
         ["每片 τ<sub>g</sub> 误差 RMS（I / Q）", "1.04 / 1.60 ps（预测 1.09）", "2.85 / 4.70 ps（预测 3.10）"],
         ["累积相位 50 GHz 处误差", "I −0.14，Q −0.06 rad（σ 0.11 / 0.14）", "I −0.09，Q +0.15 rad（σ 0.31 / 0.43）"],
         ["累积相位最大误差", "0.20 rad", "0.39 rad"],
         ["二次模型曲率（真值 −6）", "I −6.01，Q −5.56 rad", "I −6.43，Q −4.73 rad"],
         ["skew（内部 9 片，真值 0.29 ps）", "+0.04 ± 0.55 ps", "−0.64 ± 1.57 ps"]],
        [52 * mm, 54 * mm, 54 * mm]))
S(Spacer(1, 4))
S(P("误差棒由逐记录散布算出，与实测 RMS 一致；两轮之间 σ 的比值 3.0 与理论 1/ε 关系（3.2）相符，"
    "说明噪声按 √T 收敛，可以据此外推长积分时间。0.3 ps 的 skew 在 160 ms 内分辨不出，"
    "与 (7.3) 的预测一致，skew 应走 burst 路线。", body))

# ---------------------------------------------------------------- 10
S(P("10. 结论", h1))
S(P("乘性导频把“测 50 GHz 相频”变成“测一根 90 MHz 谱线的相位”。它成立的三个支柱：（1）平方律保留相干双音的相位差；"
    "（2）乘性调制让每个频率都配上一个同源副本，数据相位精确抵消，只剩 H 的相位差；"
    "（3）ν 远小于 H 的变化尺度，相位差就是群时延，逐片累加（望远镜恒等式）还原相频，I/Q 相减得到 skew。"
    "精度只由 ν、导频功率占比、被调制带宽和积分时间决定，光功率和 PD 前端不进公式。"
    "在 −25 dB 的 EVM 预算下，ν = 90 MHz 时 0.1 ps skew 需要约 5 s，0.1 rad 相频约 2 s；"
    "把 ν 提到 bin 间距 277.8 MHz 并用邻 bin MAC 实现后分别为 0.38 s 和 0.17 s。", body))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=18 * mm,
                        title="基于乘性导频的相干发射机在线 S21 监测：原理推导", author="")


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("WQY", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawCentredString(W / 2, 10 * mm, f"— {doc_.page} —")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
