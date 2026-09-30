"""Data figures (main Figures 4-7 and Supplementary Figures S1-S3) from the fixed evidence.

Run from the repository root with the analysis interpreter:
    python "QREI submission/manuscript/figure_sources/data_figures.py"
Inputs (read only): endpoint-aware features and predictions, aggregate result CSVs, raw temperature
tables and the two pairs of raw vibration records listed in comparison/spectrum_record_selection.csv.
No model is refitted and no reported number is recomputed. Style: Arial, 6-8.5 pt, bold lowercase
panel letters, thin lines, one colour per quantity throughout the paper.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from bearing_dt.qrei.signal import harmonize_record, defect_orders  # noqa: E402
from bearing_dt.qrei.summarize import NAMES  # noqa: E402

QREI = ROOT / "QREI submission"
RESULTS = QREI / "results/endpoint_v3"
FIGURES = QREI / "manuscript/figures"
SOURCES = QREI / "manuscript/figure_sources"
FEATURES = ROOT / "data/processed/phme_tvoc_10b_endpoint_v3/features.csv"
RAW = ROOT / "data/raw/phme_tvoc"

INK, MUTED, GRID = "#1f2a36", "#6b7780", "#e6e9ec"
BLUE, ORANGE, TEAL, VIOLET, RED, OLIVE = "#2f5d8a", "#c0622b", "#23857c", "#5a4a8c", "#b03a3a", "#7a8b2e"
LIGHT_BLUE = "#a9c3dc"
STOP = {"B02": "vibration stop", "B03": "temperature stop", "B04": "vibration stop", "B08": "temperature stop",
        "B10": "vibration stop", "B11": "vibration stop", "B12": "vibration stop", "B17": "vibration stop",
        "B01": "diagnostic only", "B05": "diagnostic only"}
WIDTH = 6.3  # inches; equals the manuscript text width (160 mm)


def style() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.2,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03, "savefig.dpi": 300,
    })


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV, or its gzip copy (as distributed in the public repository)."""
    gz = path.with_name(path.name + ".gz")
    return pd.read_csv(path if path.exists() else gz)


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(SOURCES / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


def letter(subfig, text: str, x=0.0, y=1.0) -> None:
    subfig.text(x, y, text, fontsize=8.5, fontweight="bold", ha="left", va="top")


def tag(ax, bearing: str, extra: str = "", short: bool = False) -> None:
    label = STOP[bearing].replace(" stop", "") if short else STOP[bearing]
    ax.set_title(r"$\mathbf{" + bearing + r"}$" + "   " + label + extra, loc="left", fontsize=6.8, pad=3)


def temperatures(bearing: str) -> pd.DataFrame:
    return pd.read_csv(RAW / bearing / bearing / f"{bearing}_meanTemperatures.csv")


def draw_vibration(ax, g):
    ax.axhspan(6, 10, color=ORANGE, alpha=0.09, lw=0)
    ax.plot(g.elapsed_hours, g.A_rms_g, color=BLUE, lw=0.45, alpha=0.55)
    ax.plot(g.elapsed_hours, g.C_rms_g, color=ORANGE, lw=0.45, alpha=0.55)
    ax.plot(g.elapsed_hours, g.hi_rms_g, color=INK, lw=0.9)
    ax.set_ylim(0, max(10.5, float(np.nanpercentile(np.r_[g.A_rms_g, g.C_rms_g], 99.8)) * 1.05))


def draw_temperature(ax, g, bearing):
    t = temperatures(bearing)
    ax.axhline(110, color=RED, lw=0.7, ls=(0, (4, 2)))
    ax.plot(g.elapsed_hours, t.iloc[:, 1], color=BLUE, lw=0.7)
    ax.plot(g.elapsed_hours, t.iloc[:, 2], color=ORANGE, lw=0.7)
    ax.plot(g.elapsed_hours, g.ep_room_temperature_C, color=MUTED, lw=0.6, ls=":")
    ax.set_ylim(15, 125)


VIB_HANDLES = [Line2D([], [], color=BLUE, lw=1, alpha=0.7, label="Channel A"),
               Line2D([], [], color=ORANGE, lw=1, alpha=0.7, label="Channel C"),
               Line2D([], [], color=INK, lw=1.1, label="Causal vibration indicator"),
               Patch(color=ORANGE, alpha=0.18, lw=0, label="6–10 g stop range")]
TEMP_HANDLES = [Line2D([], [], color=BLUE, lw=1, label="Position T1"),
                Line2D([], [], color=ORANGE, lw=1, label="Position T2"),
                Line2D([], [], color=MUTED, lw=0.9, ls=":", label="Ambient"),
                Line2D([], [], color=RED, lw=0.9, ls=(0, (4, 2)), label="110 °C stop limit")]


def spectrum(ax, selection: pd.DataFrame, bearing: str) -> None:
    rows = selection[selection.bearing_id == bearing]
    for _, row in rows.iterrows():
        folder = RAW / bearing / bearing / "vibrationData"
        raw = next(f for f in folder.glob("*.mat") if int(re.search(r"_M(\d+)", f.stem)[1]) == int(row.record_id))
        x, fs = harmonize_record(loadmat(raw))
        assert abs(fs - row.source_fs_Hz) < 1, (bearing, fs)
        sos = signal.butter(4, [6000, 10000], fs=64000, btype="bandpass", output="sos")
        env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x[0]), axis=-1))
        f, psd = signal.welch(env - env.mean(), fs=64000, window="hann", nperseg=16384, noverlap=8192)
        rotation = row.speed_rpm / 60.0
        keep = (f / rotation >= 0.3) & (f / rotation <= 24)
        early = row.selection.startswith("Early")
        ax.semilogy(f[keep] / rotation, psd[keep] * rotation, lw=0.7, color=BLUE if early else ORANGE,
                    label=("Early record" if early else "Penultimate record")
                    + f" ({row.elapsed_hours:.2f} h, {row.speed_rpm:,.0f} rpm)")
    orders = defect_orders()
    for key, ls, ha, dx in (("BSF", (0, (6, 2, 1, 2)), "right", -0.25), ("BPFO", (0, (1.2, 1.4)), "center", 0.0),
                            ("BPFI", (0, (4, 2)), "left", 0.25)):
        ax.axvline(orders[key], color=MUTED, lw=0.6, ls=ls)
        ax.text(orders[key] + dx, 1.0, key, transform=ax.get_xaxis_transform(), ha=ha, va="bottom", fontsize=6,
                color=MUTED)
    ax.set_title(r"$\mathbf{" + bearing + r"}$" + f"   channel A, {round(rows.source_fs_Hz.iloc[0] / 1000):d} kHz source",
                 loc="left", fontsize=6.8, pad=11)
    ax.set_xlabel("Envelope frequency / shaft frequency")
    ax.set_xlim(0, 24.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=6, handlelength=1.4, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.9, borderpad=0.3)


def figure_sensors(features: pd.DataFrame) -> None:
    bearings = ("B01", "B03", "B10", "B11")
    fig = plt.figure(figsize=(WIDTH, 4.9), layout="constrained")
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.02)
    for sub, quantity, handles, ylabel in ((top, "vibration", VIB_HANDLES, "RMS acceleration (g)"),
                                          (middle, "temperature", TEMP_HANDLES, "Temperature (°C)")):
        axes = sub.subplots(1, 4)
        for ax, b in zip(axes, bearings):
            g = features[features.bearing_id == b].sort_values("elapsed_hours")
            if quantity == "vibration":
                draw_vibration(ax, g)
            else:
                draw_temperature(ax, g, b)
            tag(ax, b)
            ax.set_xlabel("Elapsed time (h)")
        axes[0].set_ylabel(ylabel)
        sub.legend(handles=handles, loc="outside upper right", ncol=4)
    letter(top, "a")
    letter(middle, "b")
    selection = pd.read_csv(RESULTS / "comparison/spectrum_record_selection.csv")
    axes = bottom.subplots(1, 2, sharey=True)
    for ax, b in zip(axes, ("B02", "B10")):
        spectrum(ax, selection, b)
    axes[0].set_ylabel("Envelope power (g²/order)")
    letter(bottom, "c")
    save(fig, "sensors")


def forecast_axis(ax, pred, b, controls=True, short=False):
    g = pred[(pred.bearing_id == b) & (pred.model == "representation")].sort_values("elapsed_hours")
    ax.fill_between(g.elapsed_hours, g.lower_hours, g.upper_hours, color=LIGHT_BLUE, alpha=0.45, lw=0)
    ax.plot(g.elapsed_hours, g.pred_hours, color=BLUE, lw=0.55)
    if controls:
        for model, color, ls in (("competing_stop", ORANGE, "-"), ("time_only", VIOLET, (0, (4, 2)))):
            h = pred[(pred.bearing_id == b) & (pred.model == model)].sort_values("elapsed_hours")
            ax.plot(h.elapsed_hours, h.pred_hours, color=color, lw=0.65, ls=ls)
    ax.plot(g.elapsed_hours, g.rul_hours, color=INK, lw=1.0)
    life = float(g.observed_duration_hours.iloc[0])
    tag(ax, b, f", {life:.2f} h" if short else f", {life:.2f} h life", short=short)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Elapsed time (h)")


def figure_accuracy(pred: pd.DataFrame, learned, controls) -> None:
    per = pd.read_csv(RESULTS / "per_bearing_metrics.csv")
    summary = pd.read_csv(RESULTS / "model_summary.csv").set_index("model")
    fig = plt.figure(figsize=(WIDTH, 4.9), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 1.0], hspace=0.03)
    ax = top.subplots()
    order = list(learned) + list(controls)
    ypos = {m: i + (0.6 if i >= len(learned) else 0) for i, m in enumerate(order)}
    ax.axvspan(0.001, 0.20, color=TEAL, alpha=0.06, lw=0)
    ax.axvline(0.20, color=TEAL, lw=0.7, ls=(0, (4, 2)))
    ax.axvline(0.15, color=TEAL, lw=0.7, ls=(0, (1, 1.5)))
    ax.text(0.205, -0.95, "bearing target 0.20", color=TEAL, fontsize=6, ha="left", va="center")
    ax.text(0.145, -0.95, "mean target 0.15", color=TEAL, fontsize=6, ha="right", va="center")
    ax.axhline(len(learned) - 0.2, color=GRID, lw=0.8)
    rng = np.random.default_rng(3)
    for m in order:
        g = per[per.model == m]
        y = ypos[m]
        for _, r in g.iterrows():
            temp = r.bearing_id in ("B03", "B08")
            ax.plot(r.nMAE, y + rng.uniform(-0.17, 0.17), marker="^" if temp else "o", ms=3.4, ls="none",
                    mfc=ORANGE if temp else BLUE, mec="white", mew=0.3, alpha=0.9, zorder=3)
        mean = summary.loc[m, "nMAE"]
        ax.plot([mean, mean], [y - 0.34, y + 0.34], color=INK, lw=1.6, zorder=4, solid_capstyle="butt")
        ax.text(1.005, y, f"{mean:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5)
    ax.text(1.005, -0.95, "mean", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5,
            fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 20)
    ax.set_ylim(len(order) + 0.2, -1.4)
    ax.set_yticks([ypos[m] for m in order], [NAMES[m] for m in order])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Normalized error of each held-out bearing (log scale)")
    handles = [Line2D([], [], marker="o", ls="none", ms=4, mfc=BLUE, mec="white", label="Vibration-stopped bearing"),
               Line2D([], [], marker="^", ls="none", ms=4, mfc=ORANGE, mec="white", label="Temperature-stopped bearing"),
               Line2D([], [], color=INK, lw=1.6, label="Mean over 8 bearings")]
    top.legend(handles=handles, loc="outside upper right", ncol=3)
    letter(top, "a")
    axes = bottom.subplots(1, 4)
    for axx, b in zip(axes, ("B02", "B03", "B10", "B17")):
        forecast_axis(axx, pred, b, short=True)
    axes[0].set_ylabel("Remaining time (h)")
    handles = [Line2D([], [], color=INK, lw=1.2, label="Observed residual time"),
               Line2D([], [], color=BLUE, lw=0.9, label=NAMES["representation"]),
               Patch(color=LIGHT_BLUE, alpha=0.6, lw=0, label="Empirical 90% interval"),
               Line2D([], [], color=ORANGE, lw=0.9, label=NAMES["competing_stop"]),
               Line2D([], [], color=VIOLET, lw=0.9, ls=(0, (4, 2)), label=NAMES["time_only"])]
    bottom.legend(handles=handles, loc="outside upper right", ncol=3)
    letter(bottom, "b")
    save(fig, "accuracy")


CRITERIA = [("MAE_hours", "Error (h)"), ("nMAE", "Normalized error"), ("asymmetric_5", "Asymmetric error 5:1"),
            ("asymmetric_10", "Asymmetric error 10:1"), ("alpha_lambda_accuracy", "±20% accuracy"),
            ("coverage_error", "Coverage gap"), ("interval_score_hours", "Interval score (h)"),
            ("normalized_interval_score", "Normalized interval score"), ("prognostic_horizon_fraction", "Prognostic horizon")]


def figure_comparison(learned) -> None:
    pair = pd.read_csv(RESULTS / "comparison/paired_model_changes.csv").set_index("model")
    pair = pair.loc[[m for m in learned if m != "competing_stop"]]
    rank = pd.read_csv(RESULTS / "rank_probabilities.csv")
    keys = [c for c, _ in CRITERIA]
    assert set(rank.criterion) == set(keys), sorted(set(rank.criterion) ^ set(keys))
    pivot = rank.pivot(index="model", columns="criterion", values="P_rank_1").reindex(index=learned, columns=keys)
    fig = plt.figure(figsize=(WIDTH, 4.7), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.8, 1.3], hspace=0.03)
    ax = top.subplots()
    ax.axvspan(-0.2, 0, color=TEAL, alpha=0.07, lw=0)
    ax.axvline(0, color=INK, lw=0.6)
    for i, (m, v) in enumerate(pair.iterrows()):
        ax.plot([v.CI_low, v.CI_high], [i, i], color=BLUE, lw=1.2, solid_capstyle="butt")
        ax.plot(v.delta_v3_minus_v2, i, "o", color=BLUE, ms=3.8, mec="white", mew=0.5)
        ax.text(1.01, i, f"{v.delta_v3_minus_v2:+.3f}  [{v.CI_low:+.3f}, {v.CI_high:+.3f}]".replace("-", "−"),
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.3, color=MUTED)
    ax.set_yticks(range(len(pair)), [NAMES[m] for m in pair.index])
    ax.invert_yaxis()
    ax.set_xlim(-0.2, 0.95)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Change in mean normalized error (endpoint-aware minus vibration-only inputs)")
    ax.text(-0.1, -0.8, "better with\nendpoint histories", ha="center", va="bottom", fontsize=6, color=TEAL)
    ax.text(1.01, -0.8, "change  [95% interval]", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=6.3, color=MUTED, fontweight="bold")
    letter(top, "a")
    ax = bottom.subplots()
    data = pivot.to_numpy()
    im = ax.pcolormesh(np.arange(data.shape[1] + 1) - 0.5, np.arange(data.shape[0] + 1) - 0.5, data, cmap="Blues",
                       vmin=0, vmax=1, edgecolors="white", linewidth=1.0)
    ax.set_xlim(-0.5, data.shape[1] - 0.5)
    ax.set_ylim(data.shape[0] - 0.5, -0.5)
    labels = dict(CRITERIA)
    ax.set_xticks(range(data.shape[1]), [labels[c] for c in keys], rotation=30, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(data.shape[0]), [NAMES[m] for m in pivot.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v > 0.55 else INK, fontweight="bold" if v >= 0.7 else "normal")
    cb = bottom.colorbar(im, ax=ax, shrink=0.75, aspect=25, pad=0.01)
    cb.set_label("Fraction of resamples ranked first")
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.4, length=2)
    letter(bottom, "b")
    save(fig, "comparison")


def schematic(ax) -> None:
    """First-trigger rule on illustrative curves (not data)."""
    h = 0.24
    t = np.linspace(0, 1, 400)
    true = 1 - t
    fc = 0.78 * true + 0.16 * np.exp(-((t - 0.62) / 0.3) ** 2) + 0.06
    lb = np.maximum(fc - 0.27, 0)
    ax.plot(t, true, color=INK, lw=1.1, label="Observed residual time")
    ax.plot(t, fc, color=BLUE, lw=1.0, label="Point forecast")
    ax.plot(t, lb, color=BLUE, lw=0.9, ls=(0, (4, 2)), label="Lower interval bound")
    ax.axhline(h, color=ORANGE, lw=0.8, ls=(0, (1.2, 1.6)), label="Required lead time h")
    for curve, col, text, xt, yt, ha in ((lb, TEAL, "Lower-bound trigger:\nr ≥ h, timely; r − h unused", 0.36, 0.1, "right"),
                                         (fc, BLUE, "Point trigger:\nr < h, too late", 0.92, 0.47, "center")):
        k = int(np.argmax(curve <= h))
        tk, r = t[k], true[k]
        ax.plot([tk, tk], [0, r], color=col, lw=2.2, alpha=0.5, solid_capstyle="butt")
        ax.plot(tk, h, "o", color=col, ms=4, zorder=4)
        ax.annotate(text, (tk, h), xytext=(xt, yt), fontsize=6.2, color=col, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=1, shrinkB=2))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([0, h], ["0", "h"])
    ax.set_xlabel("Elapsed operating time")
    ax.set_ylabel("Remaining time")
    ax.legend(loc="upper right", fontsize=6.2)


def figure_maintenance() -> None:
    m = pd.read_csv(RESULTS / "maintenance_summary.csv")
    m = m[m.failure_cost_ratio == 10]
    methods = [("representation", BLUE, "o"), ("attention", TEAL, "s"), ("random_forest", OLIVE, "P"),
               ("competing_stop", ORANGE, "^"), ("competing_threshold", VIOLET, "D"), ("time_only", MUTED, "v")]
    fig = plt.figure(figsize=(WIDTH, 4.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.9, 1.1], hspace=0.03)
    left, right = top.subfigures(1, 2, width_ratios=[1.35, 1.0])
    schematic(left.subplots())
    letter(left, "a")
    right.text(0.03, 0.84, "Loss for bearing b", fontsize=7, fontweight="bold", ha="left", va="top")
    right.text(0.03, 0.70, r"$J_b = c\,\mathbf{1}\{\mathrm{no\ action\ or}\ r_b<h\} + (r_b-h)_+/L_b$", fontsize=7.5,
               ha="left", va="top")
    right.text(0.03, 0.54, "c   penalty for a late or missed action\nr$_b$  observed residual time at the action\n"
               "L$_b$  observed life of the bearing\nOne action is allowed per bearing.", fontsize=6.5, color=MUTED,
               ha="left", va="top", linespacing=1.6)
    axes = bottom.subplots(1, 3, sharey=True)
    for ax, lead in zip(axes, (0.5, 1.0, 2.0)):
        g = m[m.required_lead_hours == lead]
        ax.grid(color=GRID, lw=0.5)
        for model, color, marker in methods:
            for policy, fill in (("point", "white"), ("lower_bound", color)):
                r = g[(g.model == model) & (g.policy == policy)].iloc[0]
                ax.plot(r.too_late, r.unused_life_fraction, marker=marker, ms=4.6, color=color, mfc=fill, mew=0.8,
                        ls="none", zorder=3)
        for model, marker, color, ms in (("always_action", "*", INK, 6.5), ("never_action", "X", RED, 5.2)):
            r = g[g.model == model].iloc[0]
            ax.plot(r.too_late, r.unused_life_fraction, marker=marker, ms=ms, color=color, ls="none", zorder=3)
        ax.text(0.98, 0.98, f"h = {lead:g} h", transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.04, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Bearings with no timely action")
    axes[0].set_ylabel("Mean unused-life fraction")
    handles = [Line2D([], [], color=c, marker=mk, ls="none", ms=4.5, mfc=c, label=NAMES[n]) for n, c, mk in methods]
    handles += [Line2D([], [], color=INK, marker="*", ls="none", ms=6, label="Always act at first record"),
                Line2D([], [], color=RED, marker="X", ls="none", ms=5, label="Never act"),
                Line2D([], [], color=INK, marker="o", ls="none", ms=4.5, mfc="white", label="Open: point-forecast trigger"),
                Line2D([], [], color=INK, marker="o", ls="none", ms=4.5, mfc=INK, label="Filled: lower-bound trigger")]
    bottom.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.3)
    letter(bottom, "b")
    save(fig, "maintenance")


def supplementary(features: pd.DataFrame, pred: pd.DataFrame) -> None:
    bearings = sorted(features.bearing_id.unique())
    for quantity, name, handles, ylabel in (("vibration", "all_vibration_histories", VIB_HANDLES, "RMS (g)"),
                                            ("temperature", "all_temperature_histories", TEMP_HANDLES, "Temperature (°C)")):
        fig, axes = plt.subplots(5, 2, figsize=(WIDTH, 6.6), layout="constrained")
        for ax, b in zip(axes.flat, bearings):
            g = features[features.bearing_id == b].sort_values("elapsed_hours")
            if quantity == "vibration":
                draw_vibration(ax, g)
            else:
                draw_temperature(ax, g, b)
            tag(ax, b)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Elapsed time (h)")
        fig.legend(handles=handles, loc="outside upper center", ncol=4)
        save(fig, name)
    event = sorted(pred.bearing_id.unique())
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.3), layout="constrained")
    for ax, b in zip(axes.flat, event):
        forecast_axis(ax, pred, b, controls=False, short=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Remaining time (h)")
    handles = [Line2D([], [], color=INK, lw=1.2, label="Observed residual time"),
               Line2D([], [], color=BLUE, lw=0.9, label=NAMES["representation"]),
               Patch(color=LIGHT_BLUE, alpha=0.6, lw=0, label="Empirical 90% interval")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    save(fig, "all_forecasts")


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((QREI / "protocol_endpoint_v3.json").read_text(encoding="utf-8"))
    learned = protocol["learned_models"]
    controls = ["time_only", "elapsed_clock", "context_only", "temperature_only", "degradation", "competing_threshold"]
    assert sorted(controls) == sorted(protocol["controls"])
    features = read_table(FEATURES)
    pred = read_table(RESULTS / "joined_predictions.csv")
    figure_sensors(features)
    figure_accuracy(pred, learned, controls)
    figure_comparison(learned)
    figure_maintenance()
    supplementary(features, pred)
    print("Data figures written to", FIGURES)


if __name__ == "__main__":
    main()
