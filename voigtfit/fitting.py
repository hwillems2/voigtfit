import numpy as np
from scipy.special import wofz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def gaussian(x, cen_g = 0, sigma_g = 1, amp_g = 1):
    """
    Modelling a Gaussian curve 
    """
    return amp_g/(np.sqrt(2*np.pi)*sigma_g)*np.exp(-(x-cen_g)**2/(2*sigma_g**2))

def voigt(x, offset, amp, cen, sigma, gamma):
    """
    Modelling a Voigt curve (convolution of Gaussian and Lorentzian)
    """
    z = ((x - cen) + 1j*gamma) / (sigma * np.sqrt(2))
    return offset-amp * np.real(wofz(z))

def voigt_plus_gaussian(x, offset, amp_v, cen_v, sigma_v, gamma_v, amp_g, cen_g, sigma_g):
    """
    A model having a Voigt and an additive Gaussian component
    """
    return (
        voigt(x, offset, amp_v, cen_v, sigma_v, gamma_v) +
        gaussian(x, cen_g, sigma_g, amp_g)
    )


def fit_voigt_model(wavelength_grid, min_wl, max_wl, flux, rest_wl, target_name='Unknown', mjd_obs="N/A", plot = False):
    """
    Fits a Voigt model to a given spectral line region.
        Parameters:
        wavelength_grid: array_like
            the wavelength values
        min_wl: float or int
            the minimum wavelength where the fit starts
        max_wl: float or int
            the maximum wavelength where the fit ends
        flux: array_Like
            the flux array
        rest_wl: float
            the rest wavelength of the line to be fitted
        target_name: str
            the name of the target, if not specified, defaults to 'Unknown'
        mjd_obs: float
            the MJD observation time of the object, if not specified, defaults to 'N/A'
        plot: bool, optional
            whether to plot the results, default is False

    Returns:
        (rv, rv_error, offset, amp_v, cen_v, sigma_v, gamma_v, amp_g, cen_g, sigma_g, MSE): Tuple, where
       rv: float
            the estimated radial velocity inferred by the fit
       rv_error: float
            the estimated error in radial velocity based on the covariance of the center fit
       offset: float
            the offset in y values (typically near 1 for normalized spectra)
       amp: float
            the amplitude of the Voigt contribution
       mu: float
            the center of the Voigt profile
       sigma: float
            the sigma value of the Voigt profile (Gaussian contribution)
       gamma: float
            the gamma value of the Voigt profile (Lorentzian contribution)
       MSE_voigt: float
            the mean squared error (MSE) of the model fit
    """
    # Select wavelength interval properly
    wavelengths = np.array(wavelength_grid)
    flux = np.array(flux)
    
    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)  & (np.isfinite(wavelengths) & np.isfinite(flux))
    wavelength_interval = wavelengths[mask]
    flux_interval = flux[mask]
    
    # Set initial guess: [offset, center, sigma, gamma]
    offset_guess = 1.0
    amp_guess=0.8
    center_guess = wavelength_interval[np.argmin(flux_interval)]
    sigma_guess = 0.4
    gamma_guess = 0.3

    p0 = [offset_guess, amp_guess, center_guess, sigma_guess, gamma_guess]

    # Fit the Voigt profile
    try:
        popt, pcov = curve_fit(voigt, wavelength_interval, flux_interval, p0=p0,
                               bounds=([0.5, 0.0, min_wl, 0.05, 0.05], [1.5, 1.0, max_wl, 2.0, 2.0]), maxfev=5000)
        
        MSE_voigt = np.mean((flux_interval - voigt(wavelength_interval, *popt)) ** 2)
    except RuntimeError as e:
        print(f"Voigt fit failed for {target_name}: {e}")
        return [None]*8 #Same shape as if the fit succeeds 
         

    # Get fitted parameters
    offset, amp, mu, sigma, gamma = popt
    print(f"Fit Results (Voigt, {target_name}): Offset={offset:.3f}, Amplitude={amp:.3f}, Center={mu:.3f} Å, Sigma={sigma:.3f} Å, gamma={gamma:.3f} Å")
    
    # Calculate radial velocity
    rest_wavelength = rest_wl  # Rest-frame wavelength in nm
    c = 299792.458  # km/s
    rv = (mu - rest_wavelength) / rest_wavelength * c
    
    mu_error = np.sqrt(pcov[1, 1])
    rv_error = c * mu_error / rest_wavelength
    print(f"Estimated Radial Velocity: {rv:.2f} km/s ± {rv_error:.2f} km/s ")
    
    
    if abs(rest_wl - 410) <2:
        label='H$\\delta$'
    elif abs(rest_wl - 434) < 2:
        label = '$H\\gamma$'
    else: 
        label = f'{min_wl} - {max_wl}'
    # Plot
    if plot:
        plt.plot(wavelength_interval, flux_interval, 'b.', label='Data')
        plt.plot(wavelength_interval, voigt(wavelength_interval, *popt), 'r--', label='Voigt profile fit')
        plt.axvline(rest_wavelength, color='red', linestyle=':', label='Rest Wavelength')
        plt.axvline(mu, color='green', linestyle='--', label='Fitted Center')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.title(f'Voigt profile fit to {label} Line, {target_name}\n MJD {mjd_obs}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return (rv, rv_error, offset, amp, mu, sigma, gamma, MSE_voigt)


def fit_voigt_plus_gaussian_model(wavelength_grid, min_wl, max_wl, flux, rest_wl, target_name='Unknown', mjd_obs='N/A', plot = False):
    """
    Fits a Voigt + Gaussian model to a given spectral line region with emission feature.
    
        Parameters:
        wavelength_grid: array_like
            the wavelength values (in nm)
        min_wl: float or int
            the minimum wavelength where the fit starts
        max_wl: float or int
            the maximum wavelength where the fit ends
        flux: array_Like
            the (ideally normalized) flux array 
        rest_wl: float
            the rest wavelength of the line to be fitted
        target_name: str
            the name of the target, if not specified, defaults to 'Unknown'
        mjd_obs: float
            the MJD observation time of the object, if not specified, defaults to 'N/A'
        plot: bool, optional
            whether to plot the results, default is False

    Returns:
        (rv, rv_error, offset, amp_v, cen_v, sigma_v, gamma_v, amp_g, cen_g, sigma_g, MSE): Tuple, where
       rv: float
            the estimated radial velocity inferred by the fit
       rv_error: float
            the estimated error in radial velocity based on the covariance of the center fit
       offset: float
            the offset in y values (typically near 1 for normalized spectra)
       amp_v: float
            the amplitude of the Voigt contribution
       cen_v: float
            the center of the Voigt profile
       sigma_v: float
            the sigma value of the Voigt profile (Gaussian contribution)
       gamma_v: float
            the gamma value of the Voigt profile (Lorentzian contribution)
       amp_g: float
            the amplitude of the Gaussian
       cen_g: float
            the center of the Gaussian fit
       sigma_g: float
            the standard deviation of the Gaussian
       MSE: float
            the mean squared error (MSE) of the model fit
    """
    # Select wavelength interval properly
    wavelengths = np.array(wavelength_grid)
    flux = np.array(flux)
    
    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)  & (np.isfinite(wavelengths) & np.isfinite(flux))
    wavelength_interval = wavelengths[mask]
    flux_interval = flux[mask]
    
    # Set initial guess: [offset, center, sigma, gamma]
    offset_guess = 1.0
    amp_v_guess=0.6
    cen_v_guess = wavelength_interval[np.argmin(flux_interval)]
    sigma_v_guess = 0.4
    gamma_v_guess = 0.3
    
    amp_g_guess = 0.8
    sigma_g_guess = 0.2 
    cen_g_guess =wavelength_interval[np.argmax(flux_interval)]

    p0 = [
    offset_guess,      # offset
    amp_v_guess,       # amplitude of Voigt
    cen_v_guess,       # center of Voigt
    sigma_v_guess,     # sigma of Voigt
    gamma_v_guess,     # gamma of Voigt
    amp_g_guess,       # amplitude of Gaussian
    cen_g_guess,       # center of Gaussian
    sigma_g_guess      # sigma of Gaussian
]


    # Fit the Voigt profile
    try:
        popt, pcov = curve_fit(voigt_plus_gaussian, wavelength_interval, flux_interval, p0=p0,
                               bounds = (
    [0.5,  0.0, min_wl, 0.05, 0.01, 0.05, min_wl - 0.1, 0.01],  # lower
    [1.5,  1.0, max_wl, 1,  1,  5.0,  max_wl + 0.1, 0.5]    # upper
),
                               maxfev=5000)
        
        MSE = np.mean((flux_interval - voigt_plus_gaussian(wavelength_interval, *popt)) ** 2)
    except RuntimeError as e:
        print(f"Voigt + Gaussian fit failed for {target_name}: {e}")
        return [None]*11
          # Or handle it another way, like skipping plotting

    # Get fitted parameters
    offset, amp_v, cen_v, sigma_v, gamma_v, amp_g, cen_g, sigma_g = popt

    print(f"""Fit Results (Voigt+Gaussian, {target_name}):
    Offset={offset:.3f}, Amp_v={amp_v:.3f}, cen_v={cen_v:.3f} nm,
    sigma_v={sigma_v:.3f} nm, gamma_v={gamma_v:.3f} nm,
    amp_g={amp_g:.3f}, cen_g={cen_g:.3f} nm, sigma_g={sigma_g:.3f} nm""")


    
    # Calculate radial velocity
    rest_wavelength = rest_wl  # Rest-frame wavelength in nm
    # Speed of light in km/s
    c = 299792.458
    
    # Uncertainty on Voigt center (index 2 in popt and [2,2] in covariance matrix)
    mu_error = np.sqrt(pcov[2, 2])
    rv = (cen_v - rest_wavelength) / rest_wavelength * c
    rv_error = c * mu_error / rest_wavelength

    print(f"Estimated Radial Velocity: {rv:.2f} km/s ± {rv_error:.2f} km/s ")
    
    
    if abs(rest_wl - 410) <2:
        label='H$\\delta$'
    elif abs(rest_wl - 434) < 2:
        label = '$H\\gamma$'
    else: 
        label = f'{min_wl} - {max_wl}'
    # Plot
    if plot:
        plt.plot(wavelength_interval, flux_interval, 'b.', label='Data')
        plt.plot(wavelength_interval, voigt_plus_gaussian(wavelength_interval, *popt), 'r--', label='Voigt+Gaussian profile fit')
        plt.axvline(rest_wavelength, color='red', linestyle=':', label='Rest Wavelength')
        plt.axvline(cen_v, color='green', linestyle='--', label='Fitted Center')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.title(f'Voigt+Gaussian profile fit to {label} Line, {target_name}\nMJD {mjd_obs}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return (rv, rv_error, offset, amp_v, cen_v, sigma_v, gamma_v, amp_g, cen_g, sigma_g, MSE)