"""
Main calibration routine.
"""

import sys
import numpy as np
import logging
from scipy.interpolate import interp1d

from .wavelets import analyse_arc_signal
from .landmarks import (
    landmark_register,
    synchronise_signals,
    synchronise_calibration_last,
    robust_polyfit,
)
from .crosscorr import crosscorr_analysis, generate_spectra_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def calibrate_spectral_axes(
    npix: int,
    nfib: int,
    spectra: np.ndarray,
    variance: np.ndarray,
    pred_axis: np.ndarray,
    goodfib: np.ndarray,
    lamb_tab: np.ndarray,
    flux_tab: np.ndarray,
    size_tab: int,
    maxshift: int,
) -> tuple[np.ndarray, int]:
    """
    Calibrate the pixels of extracted arclamp spectra.

    Returns
    -------
    pixcal_dp : np.ndarray
        Calibrated pixels (NPIX+1, NFIB)
    status : int
        Status code (0 = OK)
    """
    # 1. Preamble & Ref Fibre
    # Find middle good fibre
    ref_fib = nfib // 2
    if not goodfib[ref_fib]:
        # Search outwards
        found = False
        for step in range(1, nfib // 2 + 1):
            if ref_fib + step < nfib and goodfib[ref_fib + step]:
                ref_fib += step
                found = True
                break
            if ref_fib - step >= 0 and goodfib[ref_fib - step]:
                ref_fib -= step
                found = True
                break
        if not found:
            logger.error("No good fibres found.")
            return np.zeros((npix + 1, nfib)), -1

    logger.info(f"Reference fibre: {ref_fib}")

    # Pixel centers
    # pred_axis has NPIX+1 edges.
    cen_axis = 0.5 * (pred_axis[:-1] + pred_axis[1:])

    # Process Arc List (Filter by range)
    min_wave = min(pred_axis)
    max_wave = max(pred_axis)

    mask_tab = (lamb_tab >= min_wave) & (lamb_tab <= max_wave)
    muv = lamb_tab[mask_tab]
    av = flux_tab[mask_tab]
    m = len(muv)

    # Sort
    idx = np.argsort(muv)
    muv = muv[idx]
    av = av[idx]

    # Unique check (skip duplicates)
    # Simple way: use unique
    # But we want to keep associated AV.
    # Fortran does O(N^2) check.
    # Python:
    unique_mu, unique_idx = np.unique(muv, return_index=True)
    muv = muv[unique_idx]
    av = av[unique_idx]
    m = len(muv)
    logger.info(f"Unique lines: {m}")

    # 2. Statistical Analysis
    ref_signal = spectra[:, ref_fib]
    # Replace NaNs
    ref_signal = np.nan_to_num(ref_signal)

    mn_noise, sd_noise, sigma_inpix, ares, ztol = analyse_arc_signal(ref_signal)

    logger.info(
        f"Sigma: {sigma_inpix:.2f} pix. Noise: Mean={mn_noise:.2f}, SD={sd_noise:.2f}"
    )

    # 2.1 Mask blends
    # Mask lines too close
    disp = np.abs(cen_axis[-1] - cen_axis[0]) / (npix - 1)
    arcline_sigma = sigma_inpix * disp

    mask = np.zeros(m, dtype=bool)

    # Vectorized blend check
    # Check difference between adjacent lines
    diffs = np.diff(muv)
    # Indices where gap < 3*sigma
    blend_indices = np.where(diffs < 3.0 * arcline_sigma)[0]
    logger.info(f"Blend indices: {blend_indices}")
    logger.info(f"Blend diffs: {diffs[blend_indices]}")
    logger.info(f"Blend arcline_sigma: {arcline_sigma}")

    for idx in blend_indices:
        # Check fluxes. If one is dominant (>10x), keep it.
        if av[idx] < 10.0 * av[idx + 1] and av[idx + 1] < 10.0 * av[idx]:
            mask[idx] = True
            mask[idx + 1] = True  # Mask both? Fortran masks based on complex logic.
            # "if one line of flux > 10.0 the others we assume that this profile is only effected by blending in a very small way"
            # So if one is dominant, we keep it. If comparable, mask both?
            # Fortran: IF ( ... AND AV(I) < 10.0*AV(IDX0) ) THEN MASK(I)=TRUE
            # It removes the weaker one.
            pass
        elif av[idx] >= 10.0 * av[idx + 1]:
            mask[idx + 1] = True
        else:
            mask[idx] = True

    # 3. Landmark Register
    lmr, nlm = landmark_register(spectra, npix, nfib, ~goodfib, ref_fib, ares, ztol)
    logger.info(f"Number of landmarks: {nlm}")

    # 4. Rebin Spectra (Synchronise)
    rebin_spectra = synchronise_signals(
        spectra, npix, nfib, ~goodfib, ref_fib, lmr, nlm
    )

    # 5. Combine to Template
    # Average good fibers
    # Mask saturated pixels (counts > threshold or based on bad pixels)
    # Fortran uses a count check.

    template_spectra = np.zeros(npix)
    template_mask = np.zeros(npix, dtype=bool)

    # Simple average excluding zeros/nans
    # rebin_spectra has 0.0 for bad values

    valid_counts = np.sum(rebin_spectra > 0, axis=1)
    sums = np.sum(rebin_spectra, axis=1)

    # Threshold for validity: half of good fibers
    # Fortran: IF (CNT<0.5*NGOODFIBS)
    ngoodfibs = np.sum(goodfib)

    valid_pixels = valid_counts >= 0.5 * ngoodfibs
    template_spectra[valid_pixels] = sums[valid_pixels] / valid_counts[valid_pixels]
    template_mask[~valid_pixels] = True

    # Extend mask
    np_ext = 7
    # Binary dilation could work
    from scipy.ndimage import binary_dilation

    template_mask = binary_dilation(template_mask, iterations=np_ext)
    template_spectra[template_mask] = 0.0

    # 6. Cross Correlation
    fshiftv = crosscorr_analysis(
        template_spectra,
        template_mask,
        npix,
        muv,
        av,
        mask,
        m,
        sigma_inpix,
        cen_axis,
        maxshift,
    )

    # Interpolate shifted axis
    # shift_axis[i] = cen_axis[i + shift]
    # We want to know: What is the wavelength at pixel i?
    # cen_axis maps Pixel -> Wavelength (predicted)
    # fshiftv says: Template(i) corresponds to Model(i - shift).
    # So Template at i matches Wavelength at (i - shift).
    # Corrected Wavelength(i) = cen_axis(i - shift)

    # Actually, Fortran does:
    # SHIFT_AXIS(I)=TRP(FLOAT(I)+FSHIFTV(I),NPIX,PV,CEN_AXIS)
    # PV is pixel index 1..N. CEN_AXIS is Wavelength.
    # So SHIFT_AXIS(i) = Interpolate CEN_AXIS at (i + shift).

    pixel_indices = np.arange(npix, dtype=float)
    shifted_indices = pixel_indices + fshiftv

    # Extrapolate/Interpolate
    f_interp = interp1d(
        pixel_indices,
        cen_axis,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    shift_axis = f_interp(shifted_indices)

    # 7. Identify Peaks in Template (Shifted)
    # "Locate Associative Template Spectra Peaks"
    # Find peaks in template near table lines (using shift_axis to map table lines to template pixels)

    pix_newv = np.zeros(m)
    mask2 = mask.copy()

    hw = int(np.ceil(3.0 * sigma_inpix))

    for i in range(m):
        if mask2[i]:
            continue

        # Find pixel corresponding to muv[i] in shifted template
        # shift_axis maps Pixel -> Wavelength. We want Wavelength -> Pixel.
        # Inverse interpolation

        # Or find index where shift_axis is closest to muv[i]
        # idx1 = np.searchsorted(shift_axis, muv[i])
        idx1 = np.argmin(np.abs(shift_axis - muv[i]))
        idx1 = np.clip(idx1, 0, npix - 1)

        # Search window around idx1
        start = max(0, idx1 - hw)
        end = min(npix, idx1 + hw + 1)

        if start >= end:
            mask2[i] = True
            continue

        window = template_spectra[start:end]
        if len(window) == 0:
            mask2[i] = True
            continue

        local_max_idx = np.argmax(window) + start

        # Check valid peak (neighbors)
        if local_max_idx <= start or local_max_idx >= end - 1:
            mask2[i] = True
            continue

        # Quadratic refinement
        y0 = template_spectra[local_max_idx - 1]
        y1 = template_spectra[local_max_idx]
        y2 = template_spectra[local_max_idx + 1]

        denom = 2 * y1 - y0 - y2
        if denom == 0:
            mask2[i] = True
            continue

        delta = 0.5 * (y0 - y2) / denom
        pix_new = local_max_idx + delta
        if np.isnan(pix_new):
            mask2[i] = True
            continue
        pix_newv[i] = pix_new

    # 8. Robust Cubic Fit (Pixel -> Wavelength)
    # Points: (pix_newv[i], muv[i])
    valid = ~mask2
    if np.sum(valid) < 4:
        logger.warning(f"Not enough valid points for cubic fit - {np.sum(valid)} points.")
        return np.zeros((npix + 1, nfib)), -1

    logger.info(f"Valid points: {np.sum(valid)}")

    x_pts = pix_newv[valid]
    y_pts = muv[valid]

    # 0-based pixel edges?
    # Fortran: PIXEL_EDGE_AXIS(I)=FLOAT(I)-1.0
    pixel_edges = (
        np.arange(npix + 1, dtype=float) - 0.5
    )  # 0.5 shift to match pixel centers being integers?
    # Wait. Fortran: Pixel centers are I=1..N. Edges I=1..N+1.
    # Edges: 0.0, 1.0, ... N.0.
    # Pixel 1 center: 0.5.
    # In my logic above, I used pixel_indices = 0..N-1.
    # If pix_new is in 0..N-1 frame.
    # Edges should be -0.5, 0.5, ...

    # Let's align with Fortran convention or keep consistent.
    # If pix_newv is 0-based index.
    # Edges of pixel 0 are -0.5 and 0.5.
    pixel_edges = np.arange(npix + 1, dtype=float) - 0.5

    cal_axis = np.zeros(npix + 1)

    coeffs = robust_polyfit(x_pts, y_pts, 3)
    cal_axis = np.polyval(coeffs, pixel_edges)

    # 9. Synchronise Calibration
    # Map cal_axis (Ref) to all fibers
    synchcal_axes = synchronise_calibration_last(
        cal_axis, npix, nfib, ~goodfib, ref_fib, lmr, nlm
    )

    # Transpose for output (NPIX+1, NFIB) -> (NPIX+1, NFIB) ?
    # Fortran: PIXCAL_DP(NPIX+1,NFIB) = TRANSPOSE(SYNCHCAL_AXES(NFIB,NPIX+1))
    # My python synchcal_axes is (nfib, npix+1).
    pixcal_dp = synchcal_axes.T

    return pixcal_dp, 0
