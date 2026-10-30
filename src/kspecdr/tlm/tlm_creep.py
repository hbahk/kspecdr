"""
Spatial shift of a frame's fibre traces against its tramline map (port of 2dfdr
``tlm_creep.F95``, ``MEASURE_TLM_SHIFT``).

2dfdr cross-correlates the frame with the image the tramline map was made from (normally the
fibre flat) in 16 x 16 sub-images, over integer lags around ``TLM_INIT_SHIFT``, takes the
peak of each sub-image by a quadratic in log(correlation), and returns the 5-sigma clipped
mean; if the scatter of the sub-image peaks exceeds 0.1 px the shift is set to 0.

This port keeps the structure (blocks along the dispersion, one shift per block and fibre,
clipped mean, quality cut) but measures each shift against the tramline map itself: the
pixels of a block within half a pitch of a fibre's trace are stacked by their offset from it
and fitted with the pixel-integrated Gaussian of the fibre's profile width plus a constant,
on a grid of shifts refined by a parabola. No image of the flat is needed, the faint, sky-
dominated fibres of object frames contribute with their proper weights, and the quality cut
is on the standard error of the combined shift.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .sigma_profile import pixel_gaussian

logger = logging.getLogger(__name__)


def _block_shift(
    dy: np.ndarray, val: np.ndarray, var: np.ndarray, sigma: float, grid: np.ndarray
) -> Tuple[float, float, float]:
    """
    Best shift of one stacked profile: amplitude and constant are linear at each trial shift.

    Returns (shift, error, amplitude S/N); NaN when the profile is not detected or the
    minimum lies on the edge of the grid.
    """
    w = 1.0 / var
    chi2 = np.empty(len(grid))
    amp = np.empty(len(grid))
    for k, s in enumerate(grid):
        g = pixel_gaussian(dy, s, sigma)
        # Weighted normal equations of val = a g + b
        sgg, sg1, s11 = np.sum(w * g * g), np.sum(w * g), np.sum(w)
        sgv, s1v = np.sum(w * g * val), np.sum(w * val)
        det = sgg * s11 - sg1 * sg1
        if det <= 0:
            chi2[k] = np.inf
            amp[k] = 0.0
            continue
        a = (sgv * s11 - s1v * sg1) / det
        b = (s1v * sgg - sgv * sg1) / det
        chi2[k] = np.sum(w * (val - a * g - b) ** 2)
        amp[k] = a
    k = int(np.argmin(chi2))
    if not 0 < k < len(grid) - 1 or amp[k] <= 0:
        return np.nan, np.nan, 0.0
    # Parabola through the minimum and its neighbours
    c0, c1, c2 = chi2[k - 1], chi2[k], chi2[k + 1]
    h = grid[1] - grid[0]
    curv = (c0 - 2.0 * c1 + c2) / h**2
    if curv <= 0:
        return np.nan, np.nan, 0.0
    shift = grid[k] + 0.5 * h * (c0 - c2) / (c0 - 2.0 * c1 + c2)
    red = max(c1 / max(len(val) - 3, 1), 1.0)
    err = np.sqrt(2.0 / curv * red)
    # Amplitude S/N at the best shift from the same normal equations
    g = pixel_gaussian(dy, shift, sigma)
    sgg, sg1, s11 = np.sum(w * g * g), np.sum(w * g), np.sum(w)
    var_a = s11 / max(sgg * s11 - sg1 * sg1, 1e-300)
    return float(shift), float(err), float(amp[k] / np.sqrt(var_a * red))


def _clipped_mean(
    x: np.ndarray, e: np.ndarray, nsig: float = 5.0, niter: int = 5
) -> Tuple[float, float, float, np.ndarray]:
    """Inverse-variance mean with iterative clipping (2dfdr ``CLIPPED_MEAN_RMS``).

    Returns (mean, standard error, rms of the kept values, kept mask).
    """
    keep = np.isfinite(x) & np.isfinite(e) & (e > 0)
    mean = np.nan
    for _ in range(niter):
        if keep.sum() < 2:
            break
        w = 1.0 / e[keep] ** 2
        mean = np.sum(w * x[keep]) / np.sum(w)
        # Normalised scatter: clip on |x - mean| against the larger of error and rms
        rms = np.sqrt(np.average((x[keep] - mean) ** 2, weights=w))
        new = keep & (np.abs(x - mean) < nsig * np.maximum(e, rms))
        if np.array_equal(new, keep):
            break
        keep = new
    if keep.sum() == 0:
        return np.nan, np.nan, np.nan, keep
    w = 1.0 / e[keep] ** 2
    mean = np.sum(w * x[keep]) / np.sum(w)
    rms = np.sqrt(np.average((x[keep] - mean) ** 2, weights=w))
    # Standard error, inflated when the values scatter more than their errors
    chi2 = np.sum(w * (x[keep] - mean) ** 2) / max(keep.sum() - 1, 1)
    stderr = np.sqrt(max(chi2, 1.0) / np.sum(w))
    return float(mean), float(stderr), float(rms), keep


def measure_tlm_shift(
    img: np.ndarray,
    var: np.ndarray,
    tlm: np.ndarray,
    sigma: np.ndarray,
    init_shift: float = 0.0,
    nblock: int = 16,
    search: float = 2.5,
    step: float = 0.05,
    min_snr: float = 5.0,
    max_err: float = 0.05,
    use: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Measure the rigid spatial shift of the fibre traces of a frame against a tramline map.

    Parameters
    ----------
    img, var : np.ndarray
        Frame and variance, shape (nspat, nspec); NaN marks bad pixels.
    tlm : np.ndarray
        Tramline map (0-based centres), shape (nfib, nspec).
    sigma : np.ndarray
        Profile sigma per fibre and column, shape (nfib, nspec) (``SIGMAPRF`` or
        ``MWIDTH / 2.3548``); the median per fibre is used.
    init_shift : float
        Centre of the searched range (2dfdr ``TLM_INIT_SHIFT``), in pixels.
    nblock : int
        Number of blocks along the dispersion (2dfdr: 16).
    search : float
        Half-width of the searched range, in pixels.
    step : float
        Grid step of the search before the parabolic refinement.
    min_snr : float
        Minimum amplitude S/N of a block profile to use its shift.
    max_err : float
        Maximum standard error of the combined shift; above it the measurement fails.
    use : np.ndarray, optional
        Boolean per fibre; False excludes a fibre (e.g. unused or broken fibres).

    Returns
    -------
    shift : float
        Shift to add to the tramline map (pixels), NaN if the measurement failed.
    info : dict
        ``err`` (standard error), ``rms`` and ``nused`` of the kept block shifts, ``nmeas``
        (detected block profiles), ``slope`` (fitted change of the shift per 1000 columns,
        a rotation diagnostic), and the per block and fibre ``shifts``, ``errors``, ``snr``,
        ``kept`` and block centres ``xblock``.
    """
    nspat, nspec = img.shape
    nfib = tlm.shape[0]
    use = np.ones(nfib, dtype=bool) if use is None else np.asarray(use, dtype=bool)
    sig_f = np.nanmedian(sigma, axis=1)
    srt = np.sort(tlm, axis=0)
    pitch = float(np.nanmedian(np.diff(srt, axis=0))) if nfib > 1 else 10.0
    half = max(0.5 * pitch, 2.0)
    grid = init_shift + np.arange(-search, search + 0.5 * step, step)
    window = abs(init_shift) + half  # rows gathered around each trace

    edges = np.linspace(0, nspec, nblock + 1).astype(int)
    shifts = np.full((nblock, nfib), np.nan)
    errors = np.full((nblock, nfib), np.nan)
    snr = np.zeros((nblock, nfib))
    xblock = 0.5 * (edges[1:] + edges[:-1])
    y = np.arange(nspat, dtype=float)
    for b in range(nblock):
        cols = np.arange(edges[b], edges[b + 1])
        for f in range(nfib):
            c = tlm[f, cols]
            if not use[f] or not np.all(np.isfinite(c)) or not np.isfinite(sig_f[f]):
                continue
            lo = max(int(np.floor(c.min() - window)), 0)
            hi = min(int(np.ceil(c.max() + window)), nspat - 1)
            if hi - lo < 4:
                continue
            dy = y[lo : hi + 1, None] - c[None, :]
            v = img[lo : hi + 1, cols]
            vv = var[lo : hi + 1, cols]
            # Pixels within half a pitch of the trace moved by the initial shift
            sel = (np.abs(dy - init_shift) <= half) & np.isfinite(v) & np.isfinite(vv)
            sel &= vv > 0
            if sel.sum() < 20:
                continue
            s, e, a = _block_shift(dy[sel], v[sel], vv[sel], float(sig_f[f]), grid)
            if np.isfinite(s) and a >= min_snr:
                shifts[b, f], errors[b, f], snr[b, f] = s, e, a

    x = shifts.ravel()
    e = errors.ravel()
    nmeas = int(np.isfinite(x).sum())
    mean, stderr, rms, keep = _clipped_mean(x, e)
    keep = keep.reshape(shifts.shape)
    slope = np.nan
    if keep.sum() >= 3:
        xb = np.broadcast_to(xblock[:, None], shifts.shape)[keep]
        if np.ptp(xb) > 0:
            coef = np.polyfit(xb, shifts[keep], 1, w=1.0 / errors[keep])
            slope = float(coef[0] * 1000.0)
    info = {
        "err": stderr, "rms": rms, "nused": int(keep.sum()), "nmeas": nmeas, "slope": slope,
        "shifts": shifts, "errors": errors, "snr": snr, "kept": keep, "xblock": xblock,
    }
    ok = np.isfinite(mean) and keep.sum() >= 3 and stderr <= max_err
    if not ok:
        logger.warning(
            f"TLM shift measurement failed: {keep.sum()} of {nmeas} block profiles kept, "
            f"shift {mean:+.3f} +- {stderr:.3f} px (max error {max_err})"
        )
        return np.nan, info
    logger.info(
        f"TLM shift {mean:+.3f} +- {stderr:.3f} px from {keep.sum()} of {nmeas} block "
        f"profiles (rms {rms:.3f} px, slope {slope:+.3f} px per 1000 columns)"
    )
    return mean, info
