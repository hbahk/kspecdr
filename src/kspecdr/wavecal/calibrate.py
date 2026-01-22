"""
Main calibration routine.
"""

import sys
import numpy as np
import logging
from scipy.interpolate import interp1d
from astropy.table import Table
from pathlib import Path
from typing import Optional

from .wavelets import analyse_arc_signal
from .landmarks import (
    landmark_register,
    synchronise_signals,
    synchronise_calibration_last,
    robust_polyfit,
)
from .crosscorr import crosscorr_analysis, generate_spectra_model

logger = logging.getLogger(__name__)


def find_reference_fiber(nfib: int, goodfib: np.ndarray) -> int:
    """Finds a suitable reference fiber (middlemost good fiber)."""
    ref_fib = nfib // 2
    if goodfib[ref_fib]:
        return ref_fib

    # Search outwards
    for step in range(1, nfib // 2 + 1):
        if ref_fib + step < nfib and goodfib[ref_fib + step]:
            return ref_fib + step
        if ref_fib - step >= 0 and goodfib[ref_fib - step]:
            return ref_fib - step

    logger.error("No good fibres found.")
    return -1


def extract_template_spectrum(
    spectra: np.ndarray,
    nfib: int,
    npix: int,
    goodfib: np.ndarray,
    ref_fib: int,
    cen_axis: np.ndarray,
    diagnostic: Optional[bool] = False,
    diagnostic_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Extracts a high S/N template spectrum by aligning and stacking fibers.

    Returns:
        template_spectra: 1D array
        template_mask: 1D boolean array (True = masked/bad)
        lmr: Landmark register array (shifts)
        sigma_inpix: Estimated sigma in pixels
        nlm: Number of landmarks found
    """
    # 2. Statistical Analysis
    ref_signal = spectra[:, ref_fib]
    ref_signal = np.nan_to_num(ref_signal)

    mn_noise, sd_noise, sigma_inpix, ares, ztol = analyse_arc_signal(ref_signal)
    logger.info(
        f"Sigma: {sigma_inpix:.2f} pix. Noise: Mean={mn_noise:.2f}, SD={sd_noise:.2f}"
    )

    # 3. Landmark Register
    lmr, nlm = landmark_register(spectra, npix, nfib, ~goodfib, ref_fib, ares, ztol)
    logger.info(f"Number of landmarks: {nlm}")

    if diagnostic:
        if diagnostic_dir:
            if not Path(diagnostic_dir).exists():
                Path(diagnostic_dir).mkdir(parents=True, exist_ok=True)
        else:
            diagnostic_dir = Path(".")
        np.savetxt(diagnostic_dir / "LANDMARK_REGISTER.txt", lmr, fmt="%.4f")

    # 4. Rebin Spectra (Synchronise)
    rebin_spectra = synchronise_signals(
        spectra, npix, nfib, ~goodfib, ref_fib, lmr, nlm
    )

    # 5. Combine to Template
    template_spectra = np.zeros(npix)
    template_mask = np.zeros(npix, dtype=bool)

    valid_counts = np.sum(rebin_spectra > 0, axis=1)
    sums = np.nansum(rebin_spectra, axis=1)

    ngoodfibs = np.sum(goodfib)
    valid_pixels = valid_counts >= 0.5 * ngoodfibs

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        template_spectra[valid_pixels] = sums[valid_pixels] / valid_counts[valid_pixels]

    template_mask[~valid_pixels] = True

    # Extend mask
    np_ext = 7
    from scipy.ndimage import binary_dilation

    template_mask = binary_dilation(template_mask, iterations=np_ext)
    template_spectra[template_mask] = 0.0

    if diagnostic:
        np.savetxt(
            diagnostic_dir / "TEMPLATE_SPECTRA.dat",
            np.column_stack((cen_axis, template_spectra)),
            fmt="%.4f",
        )

    return template_spectra, template_mask, lmr, sigma_inpix, nlm


def find_arc_line_matches(
    template_spectra: np.ndarray,
    template_mask: np.ndarray,
    sigma_inpix: float,
    cen_axis: np.ndarray,
    npix: int,
    muv: np.ndarray,
    av: np.ndarray,
    mask: np.ndarray,  # lamp lines mask
    maxshift: int,
    diagnostic: Optional[bool] = False,
    diagnostic_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies arc lines in the template spectrum.

    Returns:
        valid_pixels: Measured pixel positions
        valid_waves: True wavelengths
        final_mask: Boolean mask of lamp lines (True = bad/unused)
    """
    m = len(muv)

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
        diagnostic=diagnostic,
    )

    # Interpolate shifted axis
    pixel_indices = np.arange(npix, dtype=float)
    shifted_indices = pixel_indices + fshiftv

    f_interp = interp1d(
        pixel_indices,
        cen_axis,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    shift_axis = f_interp(shifted_indices)

    # 6.5 Quality Check
    disp = (cen_axis[-1] - cen_axis[0]) / (npix - 1)
    arcline_sigma = sigma_inpix * disp

    model_spectra = generate_spectra_model(
        muv, av, mask, m, arcline_sigma, cen_axis, npix
    )

    if diagnostic:
        if diagnostic_dir:
            if not Path(diagnostic_dir).exists():
                Path(diagnostic_dir).mkdir(parents=True, exist_ok=True)
        else:
            diagnostic_dir = Path(".")
        np.savetxt(
            diagnostic_dir / "MODEL_SPECTRA.dat",
            np.column_stack((cen_axis, model_spectra)),
            fmt="%.4f",
        )

    mask_badcorr = mask.copy()
    hw = int(np.ceil(3.0 * sigma_inpix))

    for i in range(m):
        if mask_badcorr[i]:
            continue

        idx0 = np.argmin(np.abs(cen_axis - muv[i]))
        idx1 = np.argmin(np.abs(shift_axis - muv[i]))
        idx1 = np.clip(idx1, 0, npix - 1)

        if idx0 - hw < 0 or idx0 + hw >= npix:
            mask_badcorr[i] = True
            continue

        if idx1 - hw < 0 or idx1 + hw >= npix:
            mask_badcorr[i] = True
            continue

        win_model = model_spectra[idx0 - hw : idx0 + hw + 1]
        if np.std(win_model) == 0:
            mask_badcorr[i] = True
            continue
        m_n = (win_model - np.mean(win_model)) / np.std(win_model)

        lmaxcor = -1.0
        for loop in range(-2, 3):
            idxl = idx1 + loop
            if idxl - hw < 0 or idxl + hw >= npix:
                continue

            win_template = template_spectra[idxl - hw : idxl + hw + 1]
            if np.std(win_template) == 0:
                val = 0.0
            else:
                t_n = (win_template - np.mean(win_template)) / np.std(win_template)
                val = np.dot(t_n, m_n) / (len(t_n) - 1)

            if val > lmaxcor:
                lmaxcor = val

        if lmaxcor < 0.5:
            mask_badcorr[i] = True

    # 7. Identify Peaks in Template (Shifted)
    pix_newv = np.zeros(m)
    mask2 = mask_badcorr.copy()

    for i in range(m):
        if mask2[i]:
            continue

        idx1 = np.argmin(np.abs(shift_axis - muv[i]))
        idx1 = np.clip(idx1, 0, npix - 1)

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

        if local_max_idx <= start or local_max_idx >= end - 1:
            mask2[i] = True
            continue

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

    valid = ~mask2
    return pix_newv[valid], muv[valid], mask2


def fit_calibration_model(
    x_pts: np.ndarray, y_pts: np.ndarray, poly_order: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a robust polynomial to the points.

    Returns:
        coeffs: Polynomial coefficients
        residuals: Residuals of the fit
        outliers: Boolean mask of outliers
    """
    if len(x_pts) < poly_order + 1:
        logger.warning(f"Not enough points for fit: {len(x_pts)}")
        return np.zeros(poly_order + 1), np.array([]), np.array([])

    # Initial Fit
    coeffs = robust_polyfit(x_pts, y_pts, poly_order)

    # Residual Analysis & Outlier Rejection
    y_fit = np.polyval(coeffs, x_pts)
    residuals = y_fit - y_pts
    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))

    outliers = np.abs(residuals - med_res) >= 3.0 * mad_res

    if np.any(outliers):
        logger.info(f"Removing {np.sum(outliers)} outliers.")
        x_clean = x_pts[~outliers]
        y_clean = y_pts[~outliers]

        if len(x_clean) < poly_order + 1:
            logger.warning("Too few points after outlier rejection.")
            return coeffs, residuals, outliers  # Return initial fit if too few

        coeffs = robust_polyfit(x_clean, y_clean, poly_order)

    return coeffs, residuals, outliers


def apply_calibration_model(
    coeffs: np.ndarray,
    npix: int,
    nfib: int,
    goodfib: np.ndarray,
    ref_fib: int,
    lmr: np.ndarray,
    nlm: int,
) -> np.ndarray:
    """
    Propagates the master calibration to all fibers using landmark shifts.
    Returns pixcal_dp (NPIX+1, NFIB).
    """
    pixel_edges = np.arange(npix + 1, dtype=float) - 0.5
    cal_axis = np.polyval(coeffs, pixel_edges)

    # 9. Synchronise Calibration
    synchcal_axes = synchronise_calibration_last(
        cal_axis, npix, nfib, ~goodfib, ref_fib, lmr, nlm
    )

    return synchcal_axes.T


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
    diagnostic: Optional[bool] = False,
    diagnostic_dir: Optional[Path] = None,
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
    ref_fib = find_reference_fiber(nfib, goodfib)
    if ref_fib == -1:
        return np.zeros((npix + 1, nfib)), -1

    logger.info(f"Reference fibre: {ref_fib}")

    # Pixel centers
    cen_axis = 0.5 * (pred_axis[:-1] + pred_axis[1:])

    # Process Arc List (Filter by range)
    min_wave = min(pred_axis)
    max_wave = max(pred_axis)

    mask_tab = (lamb_tab >= min_wave) & (lamb_tab <= max_wave)
    muv = lamb_tab[mask_tab]
    av = flux_tab[mask_tab]

    # Sort
    idx = np.argsort(muv)
    muv = muv[idx]
    av = av[idx]

    # Unique check
    unique_mu, unique_idx = np.unique(muv, return_index=True)
    muv = muv[unique_idx]
    av = av[unique_idx]
    m = len(muv)
    logger.info(f"Unique lines: {m}")

    # Mask blends (2.1)
    ref_signal = spectra[:, ref_fib]
    ref_signal = np.nan_to_num(ref_signal)
    _, _, sigma_inpix, _, _ = analyse_arc_signal(ref_signal)

    disp = np.abs(cen_axis[-1] - cen_axis[0]) / (npix - 1)
    arcline_sigma = sigma_inpix * disp

    mask = np.zeros(m, dtype=bool)
    diffs = np.diff(muv)
    blend_indices = np.where(diffs < 3.0 * arcline_sigma)[0]

    for idx in blend_indices:
        if av[idx] < 10.0 * av[idx + 1] and av[idx + 1] < 10.0 * av[idx]:
            mask[idx] = True
            mask[idx + 1] = True
        elif av[idx] >= 10.0 * av[idx + 1]:
            mask[idx + 1] = True
        else:
            mask[idx] = True

    # Extract Template
    template_spectra, template_mask, lmr, sigma_inpix, nlm = extract_template_spectrum(
        spectra, nfib, npix, goodfib, ref_fib, cen_axis, diagnostic, diagnostic_dir
    )

    # Identify Arc Lines
    x_pts, y_pts, _ = find_arc_line_matches(
        template_spectra,
        template_mask,
        sigma_inpix,
        cen_axis,
        npix,
        muv,
        av,
        mask,
        maxshift,
        diagnostic,
        diagnostic_dir,
    )

    logger.info(f"Valid points: {len(x_pts)}")
    if len(x_pts) < 4:
        logger.warning(f"Not enough valid points for cubic fit - {len(x_pts)} points.")
        return np.zeros((npix + 1, nfib)), -1

    # Fit Model
    coeffs, residuals, outliers = fit_calibration_model(x_pts, y_pts, poly_order=3)

    # Calculate stats for logging
    if len(residuals) > 0:
        med_res = np.median(residuals)
        mad_res = np.median(np.abs(residuals - med_res))
        logger.info(f"Median residual: {med_res:.4f}, MAD: {mad_res:.4f}")

        rms_res = np.sqrt(np.mean((residuals**2)[~outliers]))
        logger.info(f"RMS residual: {rms_res:.4f}")

    if diagnostic:
        if diagnostic_dir:
            if not diagnostic_dir.exists():
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
        else:
            diagnostic_dir = Path(".")
        cal_centers = np.polyval(coeffs, np.arange(npix, dtype=float))
        np.savetxt(
            diagnostic_dir / "CALIBRATED_SPECTRA.dat",
            np.column_stack((cal_centers, template_spectra)),
            fmt="%.4f",
        )
    
        # identified arc lines in x_pts, y_pts, residuals, outliers, lamps
        diag = Table({
            "x_pts": x_pts, 
            "y_pts": y_pts, 
            "residuals": residuals, 
            "outliers": outliers,
        })
        diag.write(diagnostic_dir / "identified_arcs.dat", format="ascii.fixed_width_two_line", overwrite=True)
        logger.info(f"Diagnostic file written to {diagnostic_dir / 'identified_arcs.dat'}")
        
        # global fit coefficients
        diag = Table({"coeffs": coeffs})
        diag.write(diagnostic_dir / "global_fit_coefficients.dat", format="ascii.fixed_width_two_line", overwrite=True)
        logger.info(f"Diagnostic file written to {diagnostic_dir / 'global_fit_coefficients.dat'}")

    # Apply Calibration
    pixcal_dp = apply_calibration_model(coeffs, npix, nfib, goodfib, ref_fib, lmr, nlm)

    return pixcal_dp, 0
