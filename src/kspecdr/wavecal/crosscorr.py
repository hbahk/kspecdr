"""
Cross-correlation analysis for wavelength calibration.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from astropy.io import fits

logger = logging.getLogger(__name__)


def generate_spectra_model(
    muv: np.ndarray,
    av: np.ndarray,
    mask: np.ndarray,
    m: int,
    sig: float,
    t: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Generate model spectra from line list.
    """
    s = np.zeros(n)

    # Vectorized implementation
    # t is shape (n,)
    # muv is shape (m,)

    # Filter masked lines
    valid_indices = ~mask
    mu_valid = muv[valid_indices]
    a_valid = av[valid_indices]

    if len(mu_valid) == 0:
        return s

    # For each line, add Gaussian
    # This can be slow if M*N is large.
    # Optimization: Only compute near the line.

    for mu, a in zip(mu_valid, a_valid):
        # Find range +/- 5 sigma
        idx_min = np.argmin(np.abs(t - (mu - 5 * sig)))
        idx_max = np.argmin(np.abs(t - (mu + 5 * sig)))
        

        idx_min = max(0, idx_min)
        idx_max = min(n, idx_max)

        if idx_min >= idx_max:
            continue

        x = t[idx_min:idx_max]
        z = (x - mu) / sig
        y = a * np.exp(-0.5 * z**2)
        s[idx_min:idx_max] += y

    return s


def gen_cross_corr_gram(
    s1: np.ndarray, s2: np.ndarray, n: int, hw: int, max_shift: int
) -> np.ndarray:
    """
    Generate cross-correlogram.
    s1: Template
    s2: Model
    """
    # Initialize
    n_shifts = 2 * max_shift + 1
    dump_a = np.zeros((n_shifts, n))

    # Iterate through sliding windows
    # Window: [k-hw, k+hw]
    # Shift: l in [-max_shift, max_shift]
    # S2 window: [k-hw+l, k+hw+l]

    # Pre-compute s1 stats in windows?
    # Doing it directly.

    # Valid range for k
    k1 = 1 + max_shift + hw
    k2 = n - max_shift - hw  # 1-based logic conversion needed?
    # Python 0-based:
    # indices 0..N-1.
    # window center k. range k-hw .. k+hw.
    # shifted window k+l-hw .. k+l+hw.
    # need 0 <= k+l-hw and k+l+hw < n.
    # min(k+l) >= hw -> k >= hw - l. max(l)=-max_shift -> k >= hw + max_shift.
    # max(k+l) < n - hw -> k < n - hw - l. max(l)=max_shift -> k < n - hw - max_shift.

    k_start = hw + max_shift
    k_end = n - hw - max_shift

    if k_start >= k_end:
        return dump_a

    for k in range(k_start, k_end):
        s1_win = s1[k - hw : k + hw + 1]

        # Normalize s1
        std1 = np.std(s1_win)
        if std1 == 0:
            continue
        s1_norm = (s1_win - np.mean(s1_win)) / std1

        for l_idx, l in enumerate(range(-max_shift, max_shift + 1)):
            s2_win = s2[k - hw + l : k + hw + l + 1]

            std2 = np.std(s2_win)
            if std2 == 0:
                continue
            s2_norm = (s2_win - np.mean(s2_win)) / std2

            # Correlation
            corr = np.dot(s1_norm, s2_norm) / (len(s1_norm) - 1)  # Assuming N-1
            dump_a[l_idx, k] = corr

    return dump_a


def cross_corr_greedy_quad_path_search(
    crs_cgm: np.ndarray, nrows: int, npix: int
) -> np.ndarray:
    """
    Quadratic path of maximal summed correlation through the cross-correlogram.

    Port of 2dfdr ``CrossCorrGreedyQuadPathSearch`` (calibrate_spectra.F95). Each path is
    the quadratic through ``(x0, y0)``, ``(x1, y1)``, ``(x2, y2)``, with ``x0``, ``x1``,
    ``x2`` the first, middle and last column, and ``y0``, ``y1``, ``y2`` each running over
    all ``nrows`` rows. A path scores the sum of the correlogram values >= 0.5 in the cells
    it crosses; the best of the ``nrows**3`` paths is returned, the first one in 2dfdr's
    loop order (``y0``, ``y2``, ``y1``) on ties. The search is exhaustive and deterministic.

    Instead of evaluating every path at every column, each column votes: where its
    thresholded value changes between two rows, it adds the change to all paths through
    ``(y0, y2)`` whose ``y1`` puts the column above that row edge (a difference array over
    ``y1``, summed at the end). This costs O(edges * nrows**2) rather than
    O(nrows**3 * npix).

    The nodes sit on row centres (``y = row``, cells sampled at ``floor(y + 0.5)``). 2dfdr
    places them on row edges (``y = idx - 0.5`` with ``int(y + 0.5)``), which leaves a
    constant path on the boundary between two rows and its shift half a row low.

    Parameters
    ----------
    crs_cgm : np.ndarray
        Cross-correlogram, shape ``(nrows, npix)``; row ``r`` is the shift ``r - maxshift``.
    nrows : int
        Number of rows, ``2 * maxshift + 1``.
    npix : int
        Number of columns (spectral pixels).

    Returns
    -------
    np.ndarray
        The path in row coordinates, shape ``(npix,)``.
    """
    # Lagrange basis on 2dfdr's normalised axis x = (pix - 0.5) / (npix - 0.5), pix 1-based
    x = (np.arange(npix) + 0.5) / (npix - 0.5)
    x0, x1, x2 = x[0], (npix / 2.0 - 0.5) / (npix - 0.5), x[-1]
    a0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    a1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    a2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))

    # Integer-valued weights keep the sums exact, so ties break in loop order whatever
    # the order of summation.
    good = np.nan_to_num(np.asarray(crs_cgm, dtype=float)) >= 0.5
    q = np.where(good, np.rint(np.asarray(crs_cgm, dtype=float) * 2**20), 0.0)
    rows = np.arange(nrows, dtype=float)
    score = np.zeros((nrows, nrows, nrows))  # [y0, y2, y1]

    # The first and last columns (a1 = 0) do not depend on y1
    ends = a1 <= 0
    for p in np.nonzero(ends)[0]:
        y = a0[p] * rows[:, None] + a2[p] * rows[None, :]
        r = np.floor(y + 0.5).astype(int)
        inside = (r >= 0) & (r < nrows)
        score += np.where(inside, q[np.clip(r, 0, nrows - 1), p], 0.0)[:, :, None]

    # Change of each column's value at the lower edge (y = r - 0.5) of row r
    dq = np.diff(np.vstack([np.zeros(npix), q, np.zeros(npix)]), axis=0)
    dq[:, ends] = 0.0
    r_e, p_e = np.nonzero(dq)
    w = np.broadcast_to(dq[r_e, p_e], (nrows, len(r_e))).ravel()
    inv = 1.0 / a1[p_e]
    c = (r_e - 0.5) * inv
    c0 = a0[p_e] * inv
    c2 = (a2[p_e] * inv)[None, :] * rows[:, None]
    koff = (np.arange(nrows) * (nrows + 1.0))[:, None]
    nbin = nrows * (nrows + 1)
    for i in range(nrows):
        # y1 from which the path through (i, k) is above the edge at that column
        t = (c - c0 * i) - c2
        np.ceil(t, out=t)
        np.clip(t, 0, nrows, out=t)
        t += koff
        d = np.bincount(t.astype(np.intp).ravel(), w, nbin).reshape(nrows, nrows + 1)
        score[i] += np.cumsum(d[:, :nrows], axis=1)

    if score.max() <= 0:
        logger.warning("No correlation >= 0.5 in the cross-correlogram; assuming no shift")
        return np.full(npix, (nrows - 1) / 2.0)
    i, k, j = np.unravel_index(np.argmax(score), score.shape)
    return a0 * i + a1 * j + a2 * k


def determine_saturated_lines(
    template_mask: np.ndarray,
    cen_axis: np.ndarray,
    npix: int,
    sigma_inpix: float,
    muv: np.ndarray,
    av: np.ndarray,
    mask: np.ndarray,
    m: int,
    maxshift: int,
) -> np.ndarray:
    """
    Update mask for saturated lines.
    """
    # Find saturated regions in template_mask (True values)
    # Mask is boolean array.

    # Find runs of True
    is_sat = np.concatenate(([False], template_mask, [False]))
    diff = np.diff(is_sat.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    updated_mask = mask.copy()

    for start, end in zip(starts, ends):
        # Expand by maxshift
        s = max(0, start - maxshift)
        e = min(
            npix, end + maxshift
        )  # end is exclusive in Python slice, inclusive pixel index in Fortran logic often needs care

        if s >= e:
            continue

        # Wavelength range
        # cen_axis is length npix.
        # Check boundary
        if e >= npix:
            e = npix - 1

        lam_min = cen_axis[s]
        lam_max = cen_axis[e]

        # Mask lines in this range
        in_range = (muv >= lam_min) & (muv <= lam_max)
        updated_mask[in_range] = True

    return updated_mask


def crosscorr_analysis(
    template_spectra: np.ndarray,
    template_mask: np.ndarray,
    npix: int,
    muv: np.ndarray,
    av: np.ndarray,
    mask: np.ndarray,
    m: int,
    sigma_inpix: float,
    cen_axis: np.ndarray,
    maxshift: int,
    diagnostic: bool = False,
    diagnostic_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Main cross-correlation analysis routine.
    With ``diagnostic``, the correlogram is written to ``diagnostic_dir/CrsCgm0.fits``
    (the working directory if None).
    Returns: fshiftv (fractional shifts)
    """
    # 1. Determine saturated lines
    updated_mask = determine_saturated_lines(
        template_mask, cen_axis, npix, sigma_inpix, muv, av, mask, m, maxshift
    )

    # 2. Generate model spectra
    # Sigma in Angstroms
    # Estimate dispersion
    disp = (cen_axis[-1] - cen_axis[0]) / (npix - 1)
    arcline_sigma = sigma_inpix * disp

    model_spectra = generate_spectra_model(
        muv, av, updated_mask, m, arcline_sigma, cen_axis, npix
    )

    # 3. Generate Cross-Correlogram
    # Using smallest window (HW5 in Fortran)
    # WINDOW0 = NPIX/2. HW5 = WINDOW0/2/32 = NPIX/128.
    hw = max(5, npix // 128)

    crs_cgm = gen_cross_corr_gram(template_spectra, model_spectra, npix, hw, maxshift)

    if diagnostic:
        crs_file = Path(diagnostic_dir or ".") / "CrsCgm0.fits"
        try:
            crs_file.parent.mkdir(parents=True, exist_ok=True)
            fits.writeto(crs_file, crs_cgm, overwrite=True)
            logger.info(f"Diagnostic output: {crs_file}")
        except Exception as e:
            logger.warning(f"Failed to write {crs_file}: {e}")

    # 4. Determine path
    nrows = 2 * maxshift + 1
    y_path = cross_corr_greedy_quad_path_search(crs_cgm, nrows, npix)

    # 5. Convert to shifts
    # y_path is index in 0..nrows-1.
    # shift = index - maxshift
    fshiftv = y_path - maxshift

    # Optional: GreedyQuadPerturbSearchTEST (Fortran) - Skipping for now as it seems optional/diagnostic.

    return fshiftv
