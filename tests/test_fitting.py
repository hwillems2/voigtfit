import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voigtfit import fit_voigt_plus_gaussian_model
import os


def test_voigt_fit_runs():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'sample_data_with_emission.csv')
    data = np.loadtxt(data_path, delimiter=',')
    wl, flux = np.array(data[:, 0]), np.array(data[:, 1])
    result = fit_voigt_plus_gaussian_model(wl, 409, 412, flux, rest_wl=410.17, plot=True)
    assert result[0] is not None, "RV fit failed"

test_voigt_fit_runs()    #with emission
# Load sample data
data_path = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'sample_data_without_emission.csv') #without emission
data = np.loadtxt(data_path, delimiter=',')
wavelength, flux = data[:, 0], data[:, 1]

# Run the fit
results = fit_voigt_plus_gaussian_model(wavelength, 409, 412, flux, rest_wl=410.17, plot=True)


