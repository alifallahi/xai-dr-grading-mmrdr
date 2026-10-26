"""Figures for the report. All functions save a PNG and close the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data.mmrdr import GRADE_SHORT  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, MUTED = "#1f2933", "#7b8794"
COLOURS = ["#2f6fb2", "#d9822b", "#3a9e6a", "#b83f5a", "#7a5cc2"]


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def class_distribution(counts: dict[str, list[int]], path: str | Path) -> None:
    """Grouped bars: images per grade for each split."""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    width = 0.8 / len(counts)
    for i, (name, c) in enumerate(counts.items()):
        ax.bar(np.arange(5) + i * width - 0.4 + width / 2, c, width, label=name, color=COLOURS[i % 5])
    ax.set_xticks(range(5), GRADE_SHORT)
    ax.set_ylabel("images")
    ax.legend(frameon=False)
    ax.set_title("Grade distribution by split", loc="left")
    _save(fig, path)


def confusion(cm: np.ndarray, path: str | Path, title: str) -> None:
    """Counts with row-normalised shading (recall per reference grade)."""
    cm = np.asarray(cm)
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), GRADE_SHORT, rotation=30, ha="right")
    ax.set_yticks(range(5), GRADE_SHORT)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Reference grade")
    ax.set_title(title, loc="left")
    # Cells below the diagonal are under-grading.
    for i in range(5):
        for j in range(i):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#b83f5a", lw=0.6, ls=":"))
    _save(fig, path)


def reliability(bins_before: dict, bins_after: dict, ece_before: float, ece_after: float, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3), sharey=True)
    for ax, b, e, name in ((axes[0], bins_before, ece_before, "Before"), (axes[1], bins_after, ece_after, "After temperature scaling")):
        centres = (b["edges"][:-1] + b["edges"][1:]) / 2
        ok = b["counts"] > 0
        ax.bar(centres[ok], b["accuracy"][ok], width=1 / len(centres), color=COLOURS[0], alpha=0.8, ec="white", label="accuracy")
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
        ax.set_xlabel("confidence (max probability)")
        ax.set_title(f"{name}\nECE = {e:.3f}", loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    _save(fig, path)


def risk_coverage(curves: dict[str, dict], path: str | Path, ylabel: str, marker_coverage: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["coverage"], c["risk"], color=COLOURS[i % 5], lw=1.5, label=f"{name} (AURC {c['aurc']:.3f})")
    if marker_coverage is not None:
        ax.axvline(marker_coverage, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("coverage (fraction auto-accepted)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xlim(0, 1)
    _save(fig, path)


def training_curves(history, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].plot(history["epoch"], history["train_loss"], label="train", color=COLOURS[0])
    axes[0].plot(history["epoch"], history["val_loss"], label="validation", color=COLOURS[1])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False)
    for i, k in enumerate(["val_qwk", "val_macro_f1", "val_accuracy"]):
        axes[1].plot(history["epoch"], history[k], label=k.replace("val_", ""), color=COLOURS[i + 2])
    axes[1].set_xlabel("epoch")
    axes[1].legend(frameon=False)
    _save(fig, path)


def histogram(values: np.ndarray, path: str | Path, xlabel: str, title: str, vline: float | None = None, log: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.hist(values, bins=60, color=COLOURS[0], log=log)
    if vline is not None:
        ax.axvline(vline, color="#b83f5a", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("images")
    ax.set_title(title, loc="left")
    _save(fig, path)


def image_pairs(pairs: list[tuple[np.ndarray, np.ndarray, str, str]], path: str | Path, title: str) -> None:
    n = max(1, len(pairs))
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
    axes = np.atleast_2d(axes)
    for r, (a, b, ta, tb) in enumerate(pairs):
        for c, (img, t) in enumerate(((a, ta), (b, tb))):
            axes[r, c].imshow(img)
            axes[r, c].set_title(t, fontsize=7)
            axes[r, c].axis("off")
    fig.suptitle(title, fontsize=9)
    _save(fig, path)


def explanation_panel(rows: list[dict], path: str | Path, title: str) -> None:
    """Each row: original | Grad-CAM (predicted grade) | [Grad-CAM (reference grade)] | Integrated Gradients.

    The reference-grade column is drawn when any row provides ``gradcam_ref`` (used for errors:
    where is the evidence for the grade the model should have given?).
    """
    n = max(1, len(rows))
    with_ref = any(r.get("gradcam_ref") is not None for r in rows)
    ncol = 4 if with_ref else 3
    fig, axes = plt.subplots(n, ncol, figsize=(2.4 * ncol, 2.55 * n))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"])
        axes[r, 0].set_title(row["caption"], fontsize=6.5, loc="left")
        axes[r, 1].imshow(row["image"])
        axes[r, 1].imshow(row["gradcam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[r, 1].set_title("Grad-CAM: predicted grade", fontsize=7)
        c = 2
        if with_ref:
            axes[r, 2].imshow(row["image"])
            if row.get("gradcam_ref") is not None:
                axes[r, 2].imshow(row["gradcam_ref"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[r, 2].set_title("Grad-CAM: reference grade", fontsize=7)
            c = 3
        axes[r, c].imshow(row["image"])
        axes[r, c].imshow(np.ma.masked_less(row["ig"], 0.15), cmap="hot", alpha=0.85, vmin=0, vmax=1)
        axes[r, c].set_title("Integrated Gradients (+)", fontsize=7)
        for k in range(ncol):
            axes[r, k].axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.04, hspace=0.32)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    _save(fig, path)


def decision_card(case: dict, path: str | Path) -> None:
    """Prototype of how one result could be shown to a clinician.

    Separates (1) the model's output with calibrated probabilities and a conformal set,
    (2) the explanation, and (3) the action prompt, with a standing reminder that the
    clinician's own reading of the image decides.
    """
    fig = plt.figure(figsize=(7.4, 3.0))
    ax_img = fig.add_axes([0.0, 0.05, 0.36, 0.82])
    ax_img.imshow(case["image"])
    ax_img.imshow(case["gradcam"], cmap="jet", alpha=0.35, vmin=0, vmax=1)
    ax_img.axis("off")
    ax_img.set_title("Image with Grad-CAM overlay", fontsize=8, loc="left")
    ax_bar = fig.add_axes([0.50, 0.38, 0.25, 0.48])
    probs = case["probs"]
    colours = [COLOURS[0] if k in case["set"] else "#cbd2d9" for k in range(5)]
    ax_bar.barh(range(5), probs, color=colours)
    ax_bar.set_yticks(range(5), GRADE_SHORT, fontsize=7.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("calibrated probability", fontsize=7.5)
    ax_bar.set_title("Model output (blue = 90% set)", fontsize=8, loc="left")
    status = "REVIEW: model uncertain" if case["referred"] else "Model confident (not a guarantee)"
    colour = "#b83f5a" if case["referred"] else "#3a9e6a"
    fig.text(0.40, 0.22, f"Suggested grade: {GRADE_SHORT[case['pred']]}  (p = {probs[case['pred']]:.2f})", fontsize=8.5,
             weight="bold", color=INK)
    fig.text(0.40, 0.13, status, fontsize=8.5, color=colour, weight="bold")
    fig.text(0.40, 0.01, "Research prototype. The heatmap shows what influenced the model, not a confirmed\nlesion. "
             "The grade must come from your own examination of the image.", fontsize=6.5, color=MUTED)
    fig.text(0.81, 0.86, "Reference (for this report only)", fontsize=7, color=MUTED)
    fig.text(0.81, 0.79, f"grade {GRADE_SHORT[case['grade']]}", fontsize=7.5, color=INK)
    fig.text(0.81, 0.72, "annotated lesions:", fontsize=7, color=MUTED)
    fig.text(0.81, 0.69, "\n".join(case["lesions"]) or "none", fontsize=7, color=INK, va="top", linespacing=1.4)
    _save(fig, path)


def mean_curves(curves: dict[str, np.ndarray], path: str | Path, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3))
    for i, (name, arr) in enumerate(curves.items()):
        xs = np.linspace(0, 1, arr.shape[1])
        mean, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
        ax.plot(xs, mean, color=COLOURS[i % 5], label=name)
        ax.fill_between(xs, mean - se, mean + se, color=COLOURS[i % 5], alpha=0.2)
    ax.set_xlabel("fraction of FOV cells replaced (deletion) or restored (insertion)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, path)


def sanity_curve(stages: list[str], sims: dict[str, list[float]], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3))
    xs = np.arange(len(stages) + 1)
    for i, (name, vals) in enumerate(sims.items()):
        ax.plot(xs, [1.0] + list(vals), marker="o", color=COLOURS[i % 5], label=name)
    ax.set_xticks(xs, ["trained"] + [f"+{s}" for s in stages], rotation=30, ha="right")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Spearman rank correlation\nwith trained-model map")
    ax.set_title("Cascading model randomisation", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, path)


def robustness_curves(df, metric: str, path: str | Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3))
    clean = df[df["corruption"] == "none"][metric].iloc[0]
    for i, (name, sub) in enumerate(df[df["corruption"] != "none"].groupby("corruption")):
        ax.plot([0] + list(range(1, len(sub) + 1)), [clean] + sub[metric].tolist(), marker="o", color=COLOURS[i % 5], label=name)
    ax.set_xlabel("severity level (0 = original)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, path)
