# FTIR polymer classification

Classification of three industrial polymer films from FTIR spectra using
unsupervised (distance matrix, PCA) and supervised (Gaussian naive Bayes)
methods.

Final project for AAM 562 *Spectroscopic Techniques for Complex Materials
Analysis*, University of Turkish Aeronautical Association (Fall 2023–24).

## Dataset

| Class | Sample | Spectra |
|---|---|---|
| `bant` | Packaging tape | 10 |
| `poset` | Plastic bag | 10 |
| `stretch film` | Stretch film | 10 |

30 absorbance spectra, 3997–600 cm⁻¹ (1656 points), recorded on a Bruker
FTIR spectrometer in randomised order. The spectra are included as
`data/ftir_polymers.npz` (`X`: 30 × 1656, `y`: class labels,
`wavenumbers`). Raw Bruker OPUS files can be read directly with `--data`.

## Methods

1. **Spectra:** absorbance vs wavenumber per class, overlaid and offset.
2. **Distance matrix:** 30 × 30 pairwise Euclidean distances between spectra.
3. **PCA:** projection of all spectra onto the first two principal components.
4. **Classification:** Gaussian naive Bayes with a 20 % held-out test set.
5. **Training-fraction sweep:** test accuracy over 200 stratified random
   splits for each training fraction, to check that the result does not
   depend on a single split.

## Results

| | |
|---|---|
| PCA explained variance | PC1 94.1 %, PC2 4.7 % |
| Naive Bayes, 80/20 split | 100 % (6/6 test spectra) |

| Training fraction | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| Test accuracy [%] | 93.2 ± 9.1 | 84.6 ± 23.2 | 86.8 ± 14.5 | 95.1 ± 4.0 | 95.8 ± 5.4 | 96.5 ± 6.8 |

<p align="center">
  <img src="docs/pca.png" width="48%">
  <img src="docs/distance_matrix_viridis.png" width="48%">
</p>
<p align="center">
  <img src="docs/spectra_offset.png" width="48%">
  <img src="docs/training_fraction_sweep.png" width="48%">
</p>

## Usage

```bash
pip install -r requirements.txt

python ftir_classification.py all --data data/ftir_polymers.npz      # every step
python ftir_classification.py pca --data data/ftir_polymers.npz      # one step
python ftir_classification.py all --data data/ftir_polymers.npz --out figures
```

Steps: `spectra`, `distance`, `pca`, `classify`, `sweep`, `export`, `all`.
Figures are written to `figures/` by default.

To rebuild the `.npz` from raw OPUS files named `bant.0000` … `stretch film.0009`:

```bash
python ftir_classification.py export --data path/to/opus_files/
```

## License

MIT, see [LICENSE](LICENSE).
