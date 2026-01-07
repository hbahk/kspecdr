"""
Wavelet functions for peak detection and signal analysis.
Implements Mexican Hat, Haar, and N2BSpline wavelets and convolution routines.
"""

import numpy as np
import logging
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# Constants
ROOT_2PI = 2.506628274


def mexican_hat_wavelet(t: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Calculate the Mexican Hat wavelet at points t.
    Formula: A * (1 - z^2) * exp(-0.5 * z^2)
    where z = t / sigma, A = 1 / (sqrt(2*pi) * sigma^3)
    """
    z_sq = (t / sigma) ** 2
    term = 1.0 / (ROOT_2PI * sigma**3)
    return term * (1 - z_sq) * np.exp(-0.5 * z_sq)


def wavelet_convolution(signal: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:
    """
    Perform continuous wavelet transform convolution using Mexican Hat wavelet.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    t : np.ndarray
        Time/position axis for the signal. Assumed to be regularly spaced for convolution.
    scale : float
        Wavelet scale parameter (a).

    Returns
    -------
    np.ndarray
        Convolved signal (wavelet coefficients).
    """
    # Create the wavelet kernel
    # The kernel needs to be wide enough to capture the wavelet shape.
    # Mexican hat decays quickly. +/- 5*scale is usually sufficient.
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    half_width = int(np.ceil(5.0 * scale / dt))
    if half_width < 1:
        half_width = 1

    # Kernel support in units of t
    t_kernel = np.arange(-half_width, half_width + 1) * dt

    # Calculate daughter wavelet: 1/sqrt(a) * psi((t-b)/a)
    # For convolution, b is the shift, handled by convolve operation.
    # We compute 1/sqrt(scale) * psi(t_kernel / scale)
    # However, standard CWT definition involves integral.
    # Discretized: sum( signal * wavelet * dt )

    psi = mexican_hat_wavelet(t_kernel, sigma=1.0)  # Base wavelet
    # Rescale for the daughter wavelet
    # The Fortran code: DAUGHTER_WAVELET(A,B,T) = 1.0/SQRT(A)*MOTHER_WAVELET((T-B)/A)
    # Here MOTHER_WAVELET is mexican_hat_wavelet(T) with sigma=1.0.

    # The kernel evaluated at t_kernel corresponding to (t-b)
    # Let tau = t_kernel. We want 1/sqrt(a) * psi(tau/a)

    kernel_vals = mexican_hat_wavelet(t_kernel / scale, sigma=1.0) * (
        1.0 / np.sqrt(scale)
    )

    # Multiply by dt for the integral approximation
    kernel = kernel_vals * dt

    # Convolve
    # mode='same' returns output of same length as signal
    convolved = scipy_signal.convolve(signal, kernel, mode="same")

    return convolved


def wavelet_find_res_peaks_ztol(
    signal: np.ndarray, t: np.ndarray, ztol: float
) -> np.ndarray:
    """
    Find resonant peaks in signal above zero tolerance.
    Returns indices of peaks.
    """
    n = len(signal)
    peaks = []

    in_positive_range = False
    beg_idx = 0

    for i in range(1, n - 1):  # Fortran 2..N-1 (1-based), so 1..N-2 (0-based)
        if in_positive_range:
            if signal[i] < ztol:
                # End of positive range
                in_positive_range = False
                end_idx = i - 1

                # Find max in range [beg_idx, end_idx]
                max_idx = beg_idx + np.argmax(signal[beg_idx : end_idx + 1])
                peaks.append(max_idx)
        else:
            if signal[i] >= ztol:
                in_positive_range = True
                beg_idx = i

    return np.array(peaks, dtype=int)


def wavelet_find_zero_crossings2(
    signal: np.ndarray, t: np.ndarray, peaks: np.ndarray, ztol: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find LHS and RHS zero crossings for each peak.
    """
    lhs_zc = []
    rhs_zc = []
    n = len(signal)

    for p_idx in peaks:
        if signal[p_idx] <= 0.0:
            continue

        # Find LHS zero crossing
        zero_lhs = -1.0
        for j in range(p_idx, 0, -1):  # Scan left
            if signal[j] < 0.0:
                # S(j) < 0, S(j+1) > 0 (since we started from peak>0)
                # Note: Fortran loop `DO J=PIX(I),1,-1`, if S(J)<0 then interpolate J, J+1
                j0, j1 = j, j + 1
                if j1 < n:
                    denom = signal[j1] - signal[j0]
                    if denom != 0:
                        zero_lhs = t[j0] - (t[j1] - t[j0]) / denom * signal[j0]
                break

        # Find RHS zero crossing
        zero_rhs = -1.0
        for j in range(p_idx, n):  # Scan right
            if signal[j] < 0.0:
                # S(j) < 0, S(j-1) > 0
                j0, j1 = j - 1, j
                if j0 >= 0:
                    denom = signal[j1] - signal[j0]
                    if denom != 0:
                        zero_rhs = t[j0] - (t[j1] - t[j0]) / denom * signal[j0]
                break

        if zero_lhs >= 0.0 and zero_rhs >= 0.0:
            lhs_zc.append(zero_lhs)
            rhs_zc.append(zero_rhs)

    return np.array(lhs_zc), np.array(rhs_zc)


def find_resonant_peaks2(signal: np.ndarray, t: np.ndarray, ztol: float) -> np.ndarray:
    """
    Find peaks using zero crossings logic (WAVELET_FIND_RES_PEAKS2).
    Returns peak locations (interpolated float positions).
    """
    # 1. Find integer peak indices first
    peak_indices = wavelet_find_res_peaks_ztol(signal, t, ztol)

    # 2. Find zero crossings around these peaks
    lhs, rhs = wavelet_find_zero_crossings2(signal, t, peak_indices, ztol)

    # 3. Peak location is midpoint of zero crossings
    if len(lhs) > 0:
        return 0.5 * (lhs + rhs)
    else:
        return np.array([])


def calc_medmad(data: np.ndarray) -> tuple[float, float]:
    """Calculate median and MAD of data."""
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return 0.0, 0.0

    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    return med, mad


def analyse_arc_signal(arc_sig: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Analyse arc signal to estimate noise and PSF parameters.

    Returns
    -------
    mn_noise, sd_noise, al_sigma, ares, ztol
    """
    n = len(arc_sig)

    # 1. Estimate noise
    mn_noise, sd_noise = calc_medmad(arc_sig)

    # Also recalculate sd_noise excluding median values (as per Fortran)
    # "CNT=0... IF (ABS(ARCSIG(I)-MED)==0) CYCLE..."
    # Actually just standard MAD is fine, but Fortran does a second pass.
    # We will stick to the first MAD for simplicity or implement exact if needed.
    # The Fortran second pass calculates MAD of ABS(ARCSIG - MED).
    # But calc_medmad already does MAD.
    # Fortran:
    # CALL CALC_MEDMAD(TMPV,CNT,MED,MAD) -> MN=MED, SD=MAD
    # Loop... TMPV(CNT)=ABS(ARCSIG(I)-MED)
    # CALL CALC_MEDMAD(TMPV,CNT,MED,MAD) -> SD=MED
    # So the second SD estimate is the Median of the Absolute Deviations.
    # Which IS the definition of MAD. So the first call gave MAD, the second gave Median of AD?
    # Wait. First call gives Median of signal. MAD of signal.
    # Second call computes Median of |signal - Median|. Which is MAD.
    # So SD_NOISE is indeed MAD.

    # 2. Find arc line sigma
    # Noise cutoff
    noise_cutoff = mn_noise + 3.0 * sd_noise

    # CWT with scale 1.0
    scale = 1.0
    t = np.arange(n, dtype=float)
    cwt = wavelet_convolution(arc_sig, t, scale)

    # Find zero crossings
    ztol = 0.01 * np.max(cwt)

    # We need peak indices first to find zero crossings
    peak_indices = wavelet_find_res_peaks_ztol(cwt, t, ztol)
    lhs, rhs = wavelet_find_zero_crossings2(cwt, t, peak_indices, ztol)

    widths = rhs - lhs

    # Estimate sigma from widths: (gap/2)^2 = scale^2 + sigma^2
    # sigma = sqrt( (gap/2)^2 - scale^2 )
    valid_sigmas = []
    for w in widths:
        if w > 0:
            val = (0.5 * w) ** 2 - scale**2
            if val > 0:
                valid_sigmas.append(np.sqrt(val))

    if valid_sigmas:
        al_sigma = np.median(valid_sigmas)
    else:
        al_sigma = 1.0  # Fallback

    # 3. Wavelet resonance analysis
    ares = np.sqrt(5.0) * al_sigma

    # Recalculate CWT at resonance scale to get ZTOL
    cwt_res = wavelet_convolution(arc_sig, t, ares)
    max_sig = np.max(arc_sig)
    max_cwt = np.max(cwt_res)

    if max_sig > 0:
        ztol = (3 * sd_noise) * max_cwt / max_sig
    else:
        ztol = 0.0

    return mn_noise, sd_noise, al_sigma, ares, ztol
