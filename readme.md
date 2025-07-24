# voigtfit

`voigtfit` is a Python package for fitting Voigt and Voigt + Gaussian profiles to stellar spectral lines, designed for radial velocity (RV) extraction from both emission and absorption features.

Developed for spectroscopic time-series data (e.g., VFTS targets, Villaseñor et al. 2021, https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.5348V/abstract), it includes plotting, error estimation, and test data.

## Features

- Combined Voigt + Gaussian profile modeling
- Radial velocity and uncertainty estimation
- Clean fit visualization (optional)
- Sample normalized spectra included
- Built-in test functionality

## Installation

You can install the latest version directly from GitHub:

```bash
pip install git+https://github.com/hwillems2/voigtfit.git
