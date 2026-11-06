#!/usr/bin/env python3
"""
FTIR polymer classification
===========================

Classifies three industrial polymer films from FTIR transmission spectra
with unsupervised (pairwise distances, PCA) and supervised (Gaussian naive
Bayes) methods. Final project for AAM 562 "Spectroscopic Techniques for
Complex Materials Analysis", University of Turkish Aeronautical
Association, Fall 2023-24.

Dataset
-------
3 polymer classes × 10 measurements = 30 absorbance spectra, recorded on a
Bruker FTIR spectrometer and stored as OPUS binary files:

    bant.0000 ... bant.0009                  packaging tape
    poset.0000 ... poset.0009                plastic bag
    stretch film.0000 ... stretch film.0009  stretch film

Measurement order was randomised during acquisition to avoid ordering bias.

Analysis steps
--------------
spectra   Absorbance vs wavenumber for each class (overlaid or offset).
distance  30 × 30 Euclidean distance matrix between spectra.
pca       Projection onto the first two principal components.
classify  Gaussian naive Bayes on a held-out 20 % test split.
sweep     Test accuracy vs training fraction over repeated random splits,
          to check that class separation is not an artefact of one split.

Usage
-----
    python ftir_classification.py all --data data/ftir_polymers.npz
    python ftir_classification.py pca --data data/ftir_polymers.npz
    python ftir_classification.py export --data path/to/opus_files/   # → ftir_polymers.npz

`--data` may point to a folder of OPUS files or to an exported .npz file.

Author: Şeydanur Akgün
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

CLASSES = [("bant", "Packaging tape"),
           ("poset", "Plastic bag"),
           ("stretch film", "Stretch film")]
N_PER_CLASS = 10
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ── Data ──────────────────────────────────────────────────────────────────────
def load_opus(folder):
    """
    Read the 30 OPUS files and return (X, y, wavenumbers).
    X: (30, n_points) absorbance spectra, y: class index 0/1/2.
    """
    from brukeropusreader import read_file

    X, y, wavenumbers = [], [], None
    for k, (prefix, _) in enumerate(CLASSES):
        for i in range(N_PER_CLASS):
            opus = read_file(os.path.join(folder, f"{prefix}.{i:04d}"))
            X.append(opus["AB"])
            y.append(k)
            if wavenumbers is None:
                wavenumbers = opus.get_range("AB")
    return np.array(X), np.array(y), wavenumbers


def load_data(path):
    """Load from an exported .npz file or from a folder of OPUS files."""
    if path.endswith(".npz"):
        d = np.load(path)
        return d["X"], d["y"], d["wavenumbers"]
    return load_opus(path)


# ── Analysis ──────────────────────────────────────────────────────────────────
def plot_spectra(X, y, wn, out, offset=0.0):
    """Absorbance spectra per class; `offset` stacks the repeats vertically."""
    fig, axs = plt.subplots(len(CLASSES), 1, figsize=(12, 12), sharex=True)
    for k, (_, name) in enumerate(CLASSES):
        for i, spectrum in enumerate(X[y == k]):
            axs[k].plot(wn, spectrum + i * offset, lw=0.8, alpha=0.6, label=f"{name} {i + 1}")
        axs[k].set_ylabel("Absorbance")
        axs[k].set_title(name)
        axs[k].legend(loc="upper right", fontsize=7, ncol=2)
    axs[-1].set_xlabel("Wavenumber [cm$^{-1}$]")
    axs[-1].invert_xaxis()
    plt.tight_layout()
    fname = "spectra_offset.png" if offset else "spectra.png"
    plt.savefig(os.path.join(out, fname), dpi=150)
    plt.close()


def distance_matrix(X, out, cmap="viridis"):
    """Pairwise Euclidean distance matrix; class blocks appear on the diagonal."""
    D = squareform(pdist(X, metric="euclidean"))
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Euclidean distance")
    ticks = [N_PER_CLASS * k + N_PER_CLASS / 2 - 0.5 for k in range(len(CLASSES))]
    plt.xticks(ticks, [n for _, n in CLASSES])
    plt.yticks(ticks, [n for _, n in CLASSES], rotation=90, va="center")
    plt.title("Pairwise distance matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"distance_matrix_{cmap}.png"), dpi=150)
    plt.close()
    return D


def pca_projection(X, y, out):
    """Project spectra on the first two principal components."""
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    for k, (_, name) in enumerate(CLASSES):
        plt.scatter(proj[y == k, 0], proj[y == k, 1], c=COLORS[k], alpha=0.7, label=name)
    ev = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({ev[0]:.1f} % variance)")
    plt.ylabel(f"PC2 ({ev[1]:.1f} % variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("PCA of FTIR spectra")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pca.png"), dpi=150)
    plt.close()
    print(f"PCA explained variance: PC1 {ev[0]:.1f} %, PC2 {ev[1]:.1f} %")
    return proj


def classify(X, y, test_size=0.2, seed=0, split_state=30):
    """Gaussian naive Bayes on a single held-out split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    Xtr, Xte, ytr, yte = train_test_split(X[idx], y[idx], test_size=test_size,
                                          random_state=split_state)
    t0 = time.time()
    clf = GaussianNB().fit(Xtr, ytr)
    acc = np.mean(clf.predict(Xte) == yte)
    print(f"Gaussian naive Bayes: accuracy {acc * 100:.1f} % on {len(yte)} test spectra "
          f"({time.time() - t0:.3f} s)")
    return acc


def training_fraction_sweep(X, y, out, fractions=(0.1, 0.2, 0.3, 0.5, 0.7, 0.8),
                            repeats=200, seed=0):
    """Mean ± std test accuracy vs training fraction over stratified random splits."""
    rng = np.random.RandomState(seed)
    means, stds = [], []
    for f in fractions:
        acc = []
        for _ in range(repeats):
            Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=f, stratify=y,
                                                  random_state=rng.randint(1 << 30))
            acc.append(np.mean(GaussianNB().fit(Xtr, ytr).predict(Xte) == yte))
        means.append(np.mean(acc))
        stds.append(np.std(acc))
        print(f"  train fraction {f:.1f}: accuracy {means[-1] * 100:5.1f} ± {stds[-1] * 100:4.1f} %")
    means, stds = np.array(means), np.array(stds)
    plt.figure(figsize=(6, 4))
    plt.errorbar(fractions, means * 100, yerr=stds * 100, marker="o", capsize=3)
    plt.axhline(100 / len(CLASSES), color="gray", ls="--", label="Chance level")
    plt.xlabel("Training fraction")
    plt.ylabel("Test accuracy [%]")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Naive Bayes accuracy vs training fraction ({repeats} splits each)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "training_fraction_sweep.png"), dpi=150)
    plt.close()
    return means, stds


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="FTIR polymer classification")
    ap.add_argument("step", choices=["spectra", "distance", "pca", "classify",
                                     "sweep", "export", "all"])
    ap.add_argument("--data", default="data", help="OPUS folder or .npz file")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, wn = load_data(args.data)
    print(f"Loaded {X.shape[0]} spectra × {X.shape[1]} points")

    if args.step == "export":
        path = os.path.join(args.data, "ftir_polymers.npz")
        np.savez_compressed(path, X=X, y=y, wavenumbers=wn)
        print(f"Saved {path}")
        return
    if args.step in ("spectra", "all"):
        plot_spectra(X, y, wn, args.out)
        plot_spectra(X, y, wn, args.out, offset=3.95)
    if args.step in ("distance", "all"):
        distance_matrix(X, args.out, "viridis")
        distance_matrix(X, args.out, "gray")
    if args.step in ("pca", "all"):
        pca_projection(X, y, args.out)
    if args.step in ("classify", "all"):
        classify(X, y)
    if args.step in ("sweep", "all"):
        print("Training-fraction sweep:")
        training_fraction_sweep(X, y, args.out)


if __name__ == "__main__":
    main()
