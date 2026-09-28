"""
Main calibration routine.
"""

import numpy as np
import logging
from scipy.interpolate import interp1d
from scipy.optimize import linprog
from astropy.table import Table
from pathlib import Path
from typing import Optional, Tuple

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


def _parabolic_subpix(y0: float, y1: float, y2: float) -> Optional[float]:
    # returns delta in [-1, 1] roughly, or None if degenerate
    denom = 2.0 * y1 - y0 - y2
    if denom == 0.0:
        return None
    return 0.5 * (y0 - y2) / denom


def refine_peak_gaussian_fast(
    spectrum: np.ndarray,
    idx_guess: int,
    sigma0: float,
    hw: int,
    *,
    max_iter: int = 6,
    clip_sigma: Tuple[float, float] = (0.3, 8.0),
    min_amp_snr: float = 2.0,
) -> Tuple[float, float, bool]:
    """
    Fast Gaussian(+linear background) peak refinement around idx_guess.

    Model:
        y = A * exp(-(x-x0)^2/(2*s^2)) + B + C*(x-xmean)

    Returns:
        x0_refined (float), sigma_pix (float), ok (bool)
    """
    n = spectrum.size
    start = max(0, idx_guess - hw)
    end = min(n, idx_guess + hw + 1)
    if end - start < 5:
        return float(idx_guess), sigma0, False

    x = np.arange(start, end, dtype=np.float64)
    y = spectrum[start:end].astype(np.float64, copy=False)

    # quick sanity: require a peak not at the boundary
    local_max = int(np.argmax(y)) + start
    if local_max <= start or local_max >= end - 1:
        return float(idx_guess), sigma0, False

    # initial x0 from 3-point parabola around the local maximum
    y0, y1, y2 = spectrum[local_max - 1], spectrum[local_max], spectrum[local_max + 1]
    delta = _parabolic_subpix(float(y0), float(y1), float(y2))
    if delta is None or not np.isfinite(delta):
        x0 = float(local_max)
    else:
        x0 = float(local_max) + float(delta)

    s = float(np.clip(sigma0, clip_sigma[0], clip_sigma[1]))

    # center x for better conditioning of background slope
    xmean = float(np.mean(x))
    xc = x - xmean

    # robust-ish noise estimate from window (MAD)
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med))) + 1e-12
    noise = 1.4826 * mad

    # If the peak is tiny, fitting is often unstable; bail early.
    peak_amp = float(np.max(y) - med)
    if peak_amp < min_amp_snr * noise:
        return x0, s, False

    ok = True

    for _ in range(max_iter):
        # Gaussian basis at current x0, s
        dx = x - x0
        inv_s2 = 1.0 / (s * s)
        G = np.exp(-0.5 * (dx * dx) * inv_s2)

        # Linear least squares for A,B,C in: y = A*G + B + C*xc
        # Design matrix: [G, 1, xc]
        X = np.column_stack((G, np.ones_like(G), xc))
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            ok = False
            break

        A, B, C = beta
        # If amplitude goes negative or tiny, it's usually a bad fit/line blending.
        if not np.isfinite(A) or A <= 0:
            ok = False
            break

        y_model = A * G + B + C * xc
        r = y - y_model

        # Derivatives wrt x0 and s (only 2 params)
        # dG/dx0 = G * (x - x0) / s^2
        dG_dx0 = G * (dx * inv_s2)

        # dG/ds = G * (x-x0)^2 / s^3
        dG_ds = G * ((dx * dx) / (s * s * s))

        # Jacobian of model wrt [x0, s]
        # dy/dx0 = A * dG/dx0
        # dy/ds  = A * dG/ds
        J = np.column_stack((A * dG_dx0, A * dG_ds))

        # Gauss-Newton step: solve J * step = r
        try:
            step, *_ = np.linalg.lstsq(J, r, rcond=None)
        except np.linalg.LinAlgError:
            ok = False
            break

        if not np.all(np.isfinite(step)):
            ok = False
            break

        dx0, ds = float(step[0]), float(step[1])

        # damping to avoid overshoot (helps on blended/asymmetric lines)
        dx0 = np.clip(dx0, -0.7, 0.7)
        ds = np.clip(ds, -0.5, 0.5)

        x0_new = x0 + dx0
        s_new = np.clip(s + ds, clip_sigma[0], clip_sigma[1])

        # convergence
        if abs(x0_new - x0) < 1e-4 and abs(s_new - s) < 1e-4:
            x0, s = x0_new, float(s_new)
            break

        x0, s = x0_new, float(s_new)

    # keep x0 in window bounds (optional)
    if x0 < start or x0 > end - 1:
        ok = False

    return float(x0), float(s), ok


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies arc lines in the template spectrum.

    Returns:
        valid_pixels: Measured pixel positions
        valid_waves: True wavelengths
        valid_sigmas: Per-line Gaussian sigma in pixels from the fit
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
        diagnostic_dir=diagnostic_dir,
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
    sig_newv = np.full(m, np.nan)
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

        x0_fit, s_fit, ok = refine_peak_gaussian_fast(
            template_spectra,
            idx_guess=local_max_idx,
            sigma0=sigma_inpix,
            hw=hw,
            max_iter=6,
        )

        if not ok or not np.isfinite(x0_fit):
            mask2[i] = True
            continue

        pix_newv[i] = x0_fit
        sig_newv[i] = s_fit

    valid = ~mask2
    return pix_newv[valid], muv[valid], sig_newv[valid], mask2


def l1_outliers(
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    min_dev_pix: float = 0.2,
    y_per_pix: Optional[float] = None,
) -> np.ndarray:
    """
    Lines that deviate from a least-absolute-deviation polynomial fit.

    2dfdr CALIBRATE_SPECTRAL_AXES step 8: a robust cubic (``RBST_CUBEFIT2``, an L1 fit) is
    fitted to the matched lines, and with more than four lines those with
    ``|r| >= 3 median(|r|)`` are rejected. An L1 fit follows the majority of the lines, so a
    few misidentified lines do not tilt it as they tilt a least-squares fit. It passes exactly
    through at least ``order + 1`` lines; 2dfdr keeps their zero residuals in the median, which
    with seven lines or fewer makes it zero and rejects all other lines, so here the median is
    taken over the remaining residuals. The cut is also near two sigma of the centroid
    scatter, which throws away good lines when there are few of them; lines within
    ``min_dev_pix`` of the fit are kept (not in 2dfdr).

    Parameters
    ----------
    x, y : np.ndarray
        Pixel positions and wavelengths of the matched lines.
    order : int
        Polynomial order.
    min_dev_pix : float, optional
        Deviation in pixels below which a line is never rejected.
    y_per_pix : float, optional
        Units of ``y`` per pixel for that conversion; by default the local slope of the fit
        (``y`` wavelength, ``x`` pixel). Pass 1 when ``y`` is itself in pixels.

    Returns
    -------
    np.ndarray
        Boolean mask of the rejected lines.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = len(x), order + 1
    outliers = np.zeros(n, dtype=bool)
    if n <= max(4, p):
        return outliers
    # Scaled variables keep the linear programme well conditioned
    x_scale = max(0.5 * np.ptp(x), 1.0)
    y_scale = max(np.ptp(y), 1e-12)
    t = (x - 0.5 * (x.max() + x.min())) / x_scale
    b = (y - np.median(y)) / y_scale
    a = np.vander(t, p)
    # Minimise sum(u + v) subject to a c + u - v = b, u >= 0, v >= 0
    res = linprog(
        np.concatenate([np.zeros(p), np.ones(2 * n)]),
        A_eq=np.hstack([a, np.eye(n), -np.eye(n)]),
        b_eq=b,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n),
        method="highs",
    )
    if not res.success:
        logger.warning(f"L1 fit failed ({res.message}); no outlier rejection.")
        return outliers
    coef = res.x[:p]
    resid = np.abs(a @ coef - b)
    scale = np.median(np.sort(resid)[p:])
    if scale > 0:
        outliers = resid >= 3.0 * scale
    if y_per_pix is None:  # local slope of the fit, in scaled y per pixel
        slope = np.abs(np.polyval(np.polyder(coef), t)) / x_scale
    else:
        slope = abs(float(y_per_pix)) / y_scale
    dev_pix = resid / np.where(slope > 0, slope, np.inf)
    return outliers & (dev_pix >= min_dev_pix)


def fit_calibration_model(
    x_pts: np.ndarray, y_pts: np.ndarray, poly_order: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a polynomial to the matched lines after rejecting outliers of a robust fit.

    As 2dfdr CALIBRATE_SPECTRAL_AXES steps 8-9: lines that deviate from an L1 fit are
    rejected (``l1_outliers``, only with more than four lines), then a least-squares
    polynomial is fitted to the rest. The returned residuals belong to that final fit (so an
    RMS computed from them describes the solution that is used).

    Returns:
        coeffs: Polynomial coefficients of the final fit
        residuals: Residuals of the final fit (fit - true), for all points
        outliers: Boolean mask of the points rejected before the final fit
    """
    if len(x_pts) < poly_order + 1:
        logger.warning(f"Not enough points for fit: {len(x_pts)}")
        return np.zeros(poly_order + 1), np.array([]), np.array([])

    outliers = l1_outliers(x_pts, y_pts, poly_order)
    if np.sum(~outliers) < poly_order + 1:
        logger.warning("Too few points after outlier rejection; keeping all.")
        outliers[:] = False
    if np.any(outliers):
        logger.info(f"Removing {np.sum(outliers)} outliers.")

    coeffs = np.polyfit(x_pts[~outliers], y_pts[~outliers], poly_order)
    residuals = np.polyval(coeffs, x_pts) - y_pts
    return coeffs, residuals, outliers


def fit_prior_correction(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    cen_axis: np.ndarray,
    max_order: int = 2,
    n_equiv: int = 5,
):
    """
    Fit a low-order pixel correction to the predicted wavelength axis.

    ``WAVEFIT_METHOD='PRIOR'``: a line of wavelength ``y`` is expected at the pixel ``u`` where
    the predicted axis reaches ``y`` and is found at ``x = u + dx(u)``; ``dx`` is a polynomial of
    order <= ``max_order`` (0 with fewer than 3 lines, 1 with fewer than 6). The prediction
    carries the shape of the dispersion curve, so even one or two lines give a solution; its
    accuracy is that of the prediction's shape (the Isoplane optical model is off by a pixel or
    so near the detector edges, so a cubic is better once four or more well-spread lines are
    found).
    Outliers are rejected as in ``fit_calibration_model``.

    Parameters
    ----------
    x_pts, y_pts : np.ndarray
        Measured pixel positions and true wavelengths of the matched lines.
    cen_axis : np.ndarray
        Predicted wavelength at the pixel centres 0 .. npix-1.
    max_order : int, optional
        Highest order of ``dx``.
    n_equiv : int, optional
        Degree of the polynomial that approximates the solution (for diagnostics and
        ``compute_resolution_stats``).

    Returns
    -------
    wave_at : callable
        Pixel position -> wavelength.
    coeffs : np.ndarray
        Degree-``n_equiv`` polynomial (``np.polyval`` order) close to ``wave_at`` at the pixel
        centres.
    residuals : np.ndarray
        Fit minus true wavelength at the lines.
    outliers : np.ndarray
        Lines rejected before the final fit.
    order : int
        Order of ``dx`` used.
    """
    npix = cen_axis.size
    pix = np.arange(npix, dtype=float)
    xc = 0.5 * (npix - 1)
    ascending = cen_axis[-1] > cen_axis[0]
    if ascending:
        u = np.interp(y_pts, cen_axis, pix)
    else:
        u = np.interp(y_pts, cen_axis[::-1], pix[::-1])
    d = np.asarray(x_pts, dtype=float) - u

    def fit(sel):
        n = int(sel.sum())
        order = min(max_order, 0 if n < 3 else 1 if n < 6 else max_order)
        return np.polyfit((u[sel] - xc) / xc, d[sel], order), order

    c, order = fit(np.ones(len(u), dtype=bool))
    outliers = l1_outliers((u - xc) / xc, d, order, y_per_pix=1.0)  # d is in pixels
    if np.any(outliers) and np.any(~outliers):
        c, order = fit(~outliers)

    # predicted axis on a fine grid, extrapolated linearly beyond the detector
    ug = np.linspace(-20.0, npix + 19.0, 8 * (npix + 40))
    lam_g = np.interp(ug, pix, cen_axis)
    lo, hi = ug < 0, ug > npix - 1
    lam_g[lo] = cen_axis[0] + ug[lo] * (cen_axis[1] - cen_axis[0])
    lam_g[hi] = cen_axis[-1] + (ug[hi] - (npix - 1)) * (cen_axis[-1] - cen_axis[-2])
    x_g = ug + np.polyval(c, (ug - xc) / xc)

    def wave_at(x):
        return np.interp(np.asarray(x, dtype=float), x_g, lam_g)

    residuals = wave_at(x_pts) - np.asarray(y_pts, dtype=float)
    coeffs = np.polyfit(pix, wave_at(pix), n_equiv)
    return wave_at, coeffs, residuals, outliers, order


def apply_calibration_axis(
    cal_axis: np.ndarray,
    npix: int,
    nfib: int,
    goodfib: np.ndarray,
    ref_fib: int,
    lmr: np.ndarray,
    nlm: int,
) -> np.ndarray:
    """
    Propagates the reference-fibre wavelengths at the pixel edges (NPIX+1) to all fibres
    using landmark shifts. Returns pixcal_dp (NPIX+1, NFIB).
    """
    # 9. Synchronise Calibration
    synchcal_axes = synchronise_calibration_last(
        cal_axis, npix, nfib, ~goodfib, ref_fib, lmr, nlm
    )

    return synchcal_axes.T


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
    return apply_calibration_axis(cal_axis, npix, nfib, goodfib, ref_fib, lmr, nlm)


FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548


def compute_resolution_stats(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    sigma_pix: np.ndarray,
    coeffs: np.ndarray,
    outliers: np.ndarray,
    fwhm_poly_order: int = 1,
) -> dict:
    """
    Compute spectral resolution statistics from per-line Gaussian fits.

    Uses the wavelength solution polynomial to convert per-line sigma (pixels)
    to FWHM (Angstrom) via the local dispersion at each line position.

    Parameters
    ----------
    x_pts : pixel positions of matched lines
    y_pts : wavelengths of matched lines
    sigma_pix : per-line Gaussian sigma in pixels
    coeffs : wavelength solution polynomial coefficients (np.polyval order)
    outliers : boolean mask from the wavelength fit (True = outlier)
    fwhm_poly_order : order of the FWHM(lambda) polynomial fit

    Returns
    -------
    dict
        Dictionary with keys:
        ``sigma_pix_median``, ``fwhm_angstrom_median``,
        ``resolving_power_median``, ``fwhm_poly_coeffs``, and ``per_line``.
    """
    good = ~outliers
    x_good = x_pts[good]
    y_good = y_pts[good]
    s_good = sigma_pix[good]

    # Local dispersion from derivative of the wavelength polynomial
    deriv_coeffs = np.polyder(coeffs)
    local_disp = np.abs(np.polyval(deriv_coeffs, x_good))  # Å/pixel

    sigma_ang = s_good * local_disp
    fwhm_ang = FWHM_FACTOR * sigma_ang
    resolving_power = y_good / fwhm_ang

    med_sigma_pix = float(np.median(s_good))
    med_fwhm = float(np.median(fwhm_ang))
    med_wave = float(np.median(y_good))
    med_R = float(med_wave / med_fwhm) if med_fwhm > 0 else 0.0

    # Fit FWHM(lambda) polynomial for wavelength-dependent resolution
    n_good = len(y_good)
    order = min(fwhm_poly_order, max(n_good - 1, 0))
    if n_good >= 2:
        fwhm_poly = robust_polyfit(y_good, fwhm_ang, order)
    else:
        fwhm_poly = np.array([med_fwhm])

    logger.info(
        f"Resolution: median sigma={med_sigma_pix:.3f} pix, "
        f"FWHM={med_fwhm:.3f} Å, R={med_R:.0f}"
    )
    if len(fwhm_poly) > 1:
        fwhm_blue = float(np.polyval(fwhm_poly, np.min(y_good)))
        fwhm_red = float(np.polyval(fwhm_poly, np.max(y_good)))
        logger.info(
            f"FWHM trend: {fwhm_blue:.3f} Å (blue) → {fwhm_red:.3f} Å (red)"
        )

    return {
        "sigma_pix_median": med_sigma_pix,
        "fwhm_angstrom_median": med_fwhm,
        "resolving_power_median": med_R,
        "fwhm_poly_coeffs": fwhm_poly,
        "per_line": {
            "wavelength": y_good,
            "pixel": x_good,
            "sigma_pix": s_good,
            "fwhm_angstrom": fwhm_ang,
            "resolving_power": resolving_power,
        },
    }


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
    use_blends: bool = False,
    poly_order: int = 3,
    fit_method: str = "POLY",
    prior_order: int = 2,
) -> tuple[np.ndarray, int, dict]:
    """
    Calibrate the pixels of extracted arclamp spectra.

    Parameters
    ----------
    use_blends : bool, optional
        If True, blended lines (lines closer than 3-sigma to a neighbour) are
        included in the calibration rather than being excluded.  Downstream
        quality checks (cross-correlation score, Gaussian fit quality) still
        filter out lines that cannot be reliably centroided, so only lines that
        can be individually measured will contribute to the solution.  This is
        useful in spectral regions where lines are densely packed.  Default is
        False (blended lines are excluded as before).
    poly_order : int, optional
        Order of the global polynomial (``fit_method='POLY'``).
    fit_method : str, optional
        'POLY' (default): polynomial of ``poly_order`` in pixel, as 2dfdr.
        'PRIOR': low-order pixel correction to the predicted axis ``pred_axis``
        (``fit_prior_correction``); for frames with fewer than four usable lines, where a
        cubic is not constrained.
    prior_order : int, optional
        Highest order of the pixel correction with ``fit_method='PRIOR'``.

    Returns
    -------
    pixcal_dp : np.ndarray
        Calibrated pixels (NPIX+1, NFIB)
    status : int
        Status code (0 = OK)
    resolution_info : dict
        Spectral resolution measurements from arc line fits.
        Empty dict if calibration fails.  See `compute_resolution_stats`.
    """
    # 1. Preamble & Ref Fibre
    ref_fib = find_reference_fiber(nfib, goodfib)
    if ref_fib == -1:
        return np.zeros((npix + 1, nfib)), -1, {}

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

    # Measure arc-line width from the reference fibre
    ref_signal = spectra[:, ref_fib]
    ref_signal = np.nan_to_num(ref_signal)
    _, _, sigma_inpix, _, _ = analyse_arc_signal(ref_signal)

    disp = np.abs(cen_axis[-1] - cen_axis[0]) / (npix - 1)
    arcline_sigma = sigma_inpix * disp

    mask = np.zeros(m, dtype=bool)

    if not use_blends:
        # Mask blends (2.1): exclude lines that are closer than 3-sigma to a
        # neighbour so that overlapping profiles don't distort the centroid fit.
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

        logger.info(f"Blend-masked lines: {mask.sum()} / {m}")
    else:
        logger.info(
            "use_blends=True: blend masking skipped; "
            f"all {m} lines passed to cross-correlation and peak fitting."
        )

    # Extract Template
    template_spectra, template_mask, lmr, sigma_inpix, nlm = extract_template_spectrum(
        spectra, nfib, npix, goodfib, ref_fib, cen_axis, diagnostic, diagnostic_dir
    )

    # Identify Arc Lines
    x_pts, y_pts, s_pts, _ = find_arc_line_matches(
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
    fit_method = str(fit_method).upper()
    min_points = 2 if fit_method == "PRIOR" else poly_order + 1
    if len(x_pts) < min_points:
        logger.warning(
            f"Not enough valid points for the {fit_method} fit - {len(x_pts)} points."
        )
        return np.zeros((npix + 1, nfib)), -1, {}

    # Fit Model
    pixel_edges = np.arange(npix + 1, dtype=float) - 0.5
    if fit_method == "PRIOR":
        wave_at, coeffs, residuals, outliers, order = fit_prior_correction(
            x_pts, y_pts, cen_axis, max_order=prior_order
        )
        cal_axis = wave_at(pixel_edges)
        cal_centers = wave_at(np.arange(npix, dtype=float))
        logger.info(f"PRIOR fit: order-{order} pixel correction to the predicted axis")
    else:
        coeffs, residuals, outliers = fit_calibration_model(
            x_pts, y_pts, poly_order=poly_order
        )
        cal_axis = np.polyval(coeffs, pixel_edges)
        cal_centers = np.polyval(coeffs, np.arange(npix, dtype=float))

    # Calculate stats for logging
    if len(residuals) > 0:
        med_res = np.median(residuals)
        mad_res = np.median(np.abs(residuals - med_res))
        logger.info(f"Median residual: {med_res:.4f}, MAD: {mad_res:.4f}")

        rms_res = np.sqrt(np.mean((residuals**2)[~outliers]))
        logger.info(f"RMS residual: {rms_res:.4f}")

    # Compute spectral resolution from per-line Gaussian sigmas
    resolution_info = compute_resolution_stats(
        x_pts, y_pts, s_pts, coeffs, outliers
    )

    if diagnostic:
        if diagnostic_dir:
            if not diagnostic_dir.exists():
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
        else:
            diagnostic_dir = Path(".")
        np.savetxt(
            diagnostic_dir / "CALIBRATED_SPECTRA.dat",
            np.column_stack((cal_centers, template_spectra)),
            fmt="%.4f",
        )

        # Per-line measurements including resolution
        pl = resolution_info["per_line"]
        diag = Table(
            {
                "x_pts": x_pts,
                "y_pts": y_pts,
                "sigma_pix": s_pts,
                "residuals": residuals,
                "outliers": outliers,
            }
        )
        diag.write(
            diagnostic_dir / "identified_arcs.dat",
            format="ascii.fixed_width_two_line",
            overwrite=True,
        )
        logger.info(
            f"Diagnostic file written to {diagnostic_dir / 'identified_arcs.dat'}"
        )

        # Per-line resolution (good lines only, after outlier rejection)
        res_diag = Table(
            {
                "wavelength": pl["wavelength"],
                "pixel": pl["pixel"],
                "sigma_pix": pl["sigma_pix"],
                "fwhm_angstrom": pl["fwhm_angstrom"],
                "resolving_power": pl["resolving_power"],
            }
        )
        res_diag.write(
            diagnostic_dir / "resolution_per_line.dat",
            format="ascii.fixed_width_two_line",
            overwrite=True,
        )
        logger.info(
            f"Diagnostic file written to {diagnostic_dir / 'resolution_per_line.dat'}"
        )

        # global fit coefficients
        diag = Table({"coeffs": coeffs})
        diag.write(
            diagnostic_dir / "global_fit_coefficients.dat",
            format="ascii.fixed_width_two_line",
            overwrite=True,
        )
        logger.info(
            f"Diagnostic file written to {diagnostic_dir / 'global_fit_coefficients.dat'}"
        )

    # Apply Calibration
    pixcal_dp = apply_calibration_axis(cal_axis, npix, nfib, goodfib, ref_fib, lmr, nlm)

    return pixcal_dp, 0, resolution_info
