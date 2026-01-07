"""
Landmark registration and signal synchronization.
"""

import numpy as np
import logging
from scipy.interpolate import interp1d
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from .wavelets import wavelet_convolution, wavelet_find_res_peaks_ztol

logger = logging.getLogger(__name__)


def robust_polyfit(x, y, order):
    """
    Robust polynomial fitting using RANSAC.
    """
    if len(x) < order + 1:
        # Fallback to standard polyfit if not enough points
        return np.polyfit(x, y, order)

    try:
        model = make_pipeline(
            PolynomialFeatures(order),
            RANSACRegressor(LinearRegression(), random_state=42),
        )
        model.fit(x.reshape(-1, 1), y)

        # Extract coefficients - this is a bit tricky with Pipeline/RANSAC
        # Easier to just predict
        # Or use the inlier mask to do numpy polyfit
        inlier_mask = model.named_steps["ransacregressor"].inlier_mask_
        if np.sum(inlier_mask) < order + 1:
            return np.polyfit(x, y, order)

        return np.polyfit(x[inlier_mask], y[inlier_mask], order)
    except Exception:
        return np.polyfit(x, y, order)


def landmark_register(
    spectra: np.ndarray,
    npix: int,
    nfib: int,
    maskv: np.ndarray,
    ref_fib: int,
    scale: float,
    ztol: float,
) -> tuple[np.ndarray, int]:
    """
    Register and align landmarks.

    Returns
    -------
    lmr : np.ndarray (nfib, npix) - actually (nfib, nlm) but Fortran uses NPIX dimension
    nlm : int
    """
    # 1. Find landmarks in each fibre
    t = np.arange(npix, dtype=float)

    # Store peak lists
    # peaks_list[fib] = [p1, p2, ...]
    peaks_list = []

    for fib in range(nfib):
        if maskv[fib]:  # masked
            peaks_list.append([])
            continue

        signal = spectra[:, fib]
        # Replace bad values
        # Assumed handled by caller or pre-filled

        cwt = wavelet_convolution(signal, t, scale)

        # Use ztol or auto
        _ztol = ztol
        if _ztol <= 0:
            _ztol = 0.1 * np.max(cwt)

        pks = wavelet_find_res_peaks_ztol(cwt, t, _ztol)
        peaks_list.append(pks)

    # 2. Multi-Target Tracking (Simplified)
    # We want to link peaks across fibers.
    # Simple approach: Find nearest neighbors.

    # Reference peaks
    ref_peaks = peaks_list[ref_fib]

    # Structure to hold matched peaks: matches[ref_peak_idx][fib] = peak_pos
    matches = {i: {ref_fib: p} for i, p in enumerate(ref_peaks)}

    # For each fiber, try to match to ref peaks
    # We can assume distortion is smooth.
    # Start from ref_fib and go up/down.

    for direction in [1, -1]:
        curr_fib = ref_fib
        while True:
            next_fib = curr_fib + direction
            if next_fib < 0 or next_fib >= nfib:
                break

            if maskv[next_fib]:
                curr_fib = next_fib
                continue

            curr_peaks = peaks_list[next_fib]

            # Match existing tracks to curr_peaks
            for ref_idx in list(matches.keys()):
                # Get position in previous valid fiber (or closest)
                # Ideally track prediction. Here simple nearest.
                # Find last valid position for this track
                last_pos = None
                # Search backwards from next_fib
                for f in range(next_fib - direction, ref_fib - direction, -direction):
                    if f in matches[ref_idx]:
                        last_pos = matches[ref_idx][f]
                        break

                if last_pos is None:
                    continue

                # Find nearest peak in curr_peaks
                if len(curr_peaks) == 0:
                    continue

                dists = np.abs(curr_peaks - last_pos)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]

                # Threshold (e.g. 20 pixels)
                if min_dist < 20.0:
                    matches[ref_idx][next_fib] = curr_peaks[min_idx]

            curr_fib = next_fib

    # 3. Compile LMR
    # Select tracks present in > 50% of fibers
    # Or > 3 fibers

    valid_tracks = []
    for ref_idx, track in matches.items():
        if len(track) > 3:  # Arbitrary threshold
            valid_tracks.append(track)

    nlm = len(valid_tracks)
    lmr = np.zeros((nfib, nlm))

    for i, track in enumerate(valid_tracks):
        for fib, pos in track.items():
            lmr[fib, i] = pos

    return lmr, nlm


def synchronise_signals(
    spectra: np.ndarray,
    npix: int,
    nfib: int,
    maskv: np.ndarray,
    ref_fib: int,
    lmr: np.ndarray,
    nlm: int,
) -> np.ndarray:
    """
    Rebin spectra to align landmarks.
    """
    rebin_spectra = np.zeros_like(spectra)

    axis1 = np.arange(npix, dtype=float)  # Reference axis

    for fib in range(nfib):
        if maskv[fib]:
            continue

        # Get landmarks
        x_pts = []  # In this fibre
        y_pts = []  # In ref fibre

        for i in range(nlm):
            p_fib = lmr[fib, i]
            p_ref = lmr[ref_fib, i]
            if p_fib > 0 and p_ref > 0:
                x_pts.append(p_fib)
                y_pts.append(p_ref)

        if len(x_pts) < 3:
            rebin_spectra[:, fib] = spectra[:, fib]
            continue

        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)

        # Fit mapping: ref_pos = f(fib_pos).
        # We want to rebin current spectra (on fib_pos) to ref_pos grid.
        # New axis for this fiber: where do the pixels map to on the ref axis?
        # axis2 = f(axis1)

        # Normalize for stability
        coeffs = robust_polyfit(x_pts / npix, y_pts / npix, 2)
        axis2_norm = np.polyval(coeffs, axis1 / npix)
        axis2 = axis2_norm * npix

        # Interpolate
        # We have data on axis1 (0..N-1). We want data on axis1 in the new frame?
        # Rebinning usually means: Spect[new_x] = Spect[old_x].
        # If axis2 maps old pixel to new pixel coords.
        # We want rebin_spectra[x] = spectra[inverse_map(x)]
        # Or: rebin_spectra at grid 'axis1' corresponds to spectra at 'inverse_map(axis1)'

        # Here `axis2` represents the position in the reference frame corresponding to `axis1` in the current frame.
        # So `axis2` are the "true" coordinates of the pixels `axis1`.
        # We want to resample `spectra` (at coords `axis2`) onto the grid `axis1`.
        # So we interpolating (axis2, spectra) onto (axis1).

        f_interp = interp1d(
            axis2, spectra[:, fib], kind="linear", bounds_error=False, fill_value=0.0
        )
        rebin_spectra[:, fib] = f_interp(axis1)

    return rebin_spectra


def synchronise_calibration_last(
    cal_axis: np.ndarray,
    npix: int,
    nfib: int,
    maskv: np.ndarray,
    ref_fib: int,
    lmr: np.ndarray,
    nlm: int,
) -> np.ndarray:
    """
    Synchronise calibration from ref fibre to others.

    cal_axis: Calibration of reference fibre (wavelengths).
    """
    synchcal_axes = np.zeros((nfib, npix + 1))

    # cal_axis has length NPIX+1 (edges)
    axis1 = np.arange(npix + 1, dtype=float) - 0.5  # Pixel edges?
    # In Fortran: AXIS1(I)=FLOAT(I)-1.0. For I=1..NPIX+1. So 0.0 to NPIX.0.
    axis1 = np.arange(npix + 1, dtype=float)

    for fib in range(nfib):
        if maskv[fib]:
            continue

        x_pts = []  # In this fibre (pixel)
        y_pts = []  # In ref fibre (pixel)

        for i in range(nlm):
            p_fib = lmr[fib, i]
            p_ref = lmr[ref_fib, i]
            if p_fib > 0 and p_ref > 0:
                x_pts.append(p_fib)
                y_pts.append(p_ref)

        if len(x_pts) < 3:
            synchcal_axes[fib, :] = cal_axis  # Fallback
            continue

        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)

        # Map this fibre pixels -> ref fibre pixels
        coeffs = robust_polyfit(x_pts / npix, y_pts / npix, 3)  # Cubic

        axis1_norm = axis1 / npix
        axis2_norm = np.polyval(coeffs, axis1_norm)
        axis2 = axis2_norm * npix

        # axis2 contains coordinates in Ref Fibre Pixels.
        # We know Ref Fibre Pixels -> Wavelength (cal_axis).
        # Interpolate Wavelength at axis2.

        f_interp = interp1d(
            axis1, cal_axis, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        synchcal_axes[fib, :] = f_interp(axis2)

    return synchcal_axes
