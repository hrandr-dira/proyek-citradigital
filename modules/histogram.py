"""
modules/histogram.py
Menampilkan histogram dari gambar input (grayscale & RGB)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_histogram(image: np.ndarray) -> Figure:
    """
    Membuat histogram dari gambar.
    - Grayscale: 1 plot intensitas
    - RGB: 3 plot per channel (R, G, B) + 1 gabungan
    """
    fig, axes = plt.subplots(1, 1, figsize=(7, 3.5))
    fig.patch.set_facecolor("#0e1117")
    axes.set_facecolor("#1a1f2e")

    if len(image.shape) == 2:
        # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        axes.plot(hist, color="#00d4ff", linewidth=1.5)
        axes.fill_between(range(256), hist.flatten(), alpha=0.3, color="#00d4ff")
        axes.set_title("Histogram Grayscale", color="white", fontsize=11)
    else:
        # RGB — plot 3 channel dengan warna berbeda
        colors = [("#ff4d4d", "Red"), ("#4dff88", "Green"), ("#4d9fff", "Blue")]
        for i, (color, label) in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            axes.plot(hist, color=color, linewidth=1.5, label=label, alpha=0.85)
            axes.fill_between(range(256), hist.flatten(), alpha=0.15, color=color)
        axes.legend(facecolor="#1a1f2e", labelcolor="white", fontsize=9)
        axes.set_title("Histogram RGB", color="white", fontsize=11)

    axes.set_xlabel("Intensitas Piksel (0–255)", color="#aaaaaa", fontsize=9)
    axes.set_ylabel("Frekuensi", color="#aaaaaa", fontsize=9)
    axes.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in axes.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout(pad=1.5)
    return fig
