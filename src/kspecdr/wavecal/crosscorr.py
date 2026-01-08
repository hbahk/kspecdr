"""
Cross-correlation analysis for wavelength calibration.
"""

import numpy as np
import logging
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
    Greedy quadratic path search through cross-correlogram.
    """
    # nrows = 2*max_shift+1
    # We define quadratics by 3 points (x0, y0), (x1, y1), (x2, y2)
    # x0, x1, x2 fixed at start, mid, end columns.
    # y0, y1, y2 iterate through rows.

    x0 = 0.0
    x1 = npix / 2.0
    x2 = float(npix)

    x = np.arange(npix, dtype=float)

    # Precompute Lagrange polynomials
    # L0 = (x-x1)(x-x2) / (x0-x1)(x0-x2)
    l0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    l1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    l2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))

    max_sum = -1.0
    best_path = np.zeros(npix)

    # Subsampling to speed up? Fortran code does full search.
    # Assuming nrows is small (~100). 100^3 = 1e6 iterations.
    # Vectorized summing is fast.

    rows = np.arange(nrows)

    # Iterate mid first?
    # Python loops are slow. We should try to vectorize or reduce search space.
    # Or use scipy optimizer?

    # Let's implement the brute force but maybe with coarser steps if needed.

    # To optimize: Sum( CrsCgm(Path(x), x) )
    # Path(x) = y0*L0 + y1*L1 + y2*L2

    # Let's try `scipy.optimize`?
    # Function to minimize: -Sum( CrsCgm(y(x), x) )
    # Vars: y0, y1, y2. Bounds: [0, nrows-1].

    def objective(params):
        y0, y1, y2 = params
        y_path = y0 * l0 + y1 * l1 + y2 * l2

        # Sample CrsCgm
        y_indices = np.clip(np.round(y_path).astype(int), 0, nrows - 1)

        # Sum values > 0.5
        vals = crs_cgm[y_indices, np.arange(npix)]
        score = np.sum(vals[vals > 0.5])
        return -score

    # Global optimization might be needed. Or Basin Hopping.
    # Given the discrete nature and "ridge" finding, simple gradient descent might fail if not close.
    # Fortran does brute force.

    # Let's try Differential Evolution or SHGO as requested/approved.
    from scipy.optimize import differential_evolution

    bounds = [(0, nrows - 1), (0, nrows - 1), (0, nrows - 1)]
    result = differential_evolution(objective, bounds, maxiter=20, popsize=10, tol=0.01)

    y0, y1, y2 = result.x
    best_path = y0 * l0 + y1 * l1 + y2 * l2

    return best_path


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
) -> np.ndarray:
    """
    Main cross-correlation analysis routine.
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
        try:
            fits.writeto("CrsCgm0.fits", crs_cgm, overwrite=True)
            logger.info("Diagnostic output: CrsCgm0.fits")
        except Exception as e:
            logger.warning(f"Failed to write CrsCgm0.fits: {e}")

    # 4. Determine path
    nrows = 2 * maxshift + 1
    y_path = cross_corr_greedy_quad_path_search(crs_cgm, nrows, npix)

    # 5. Convert to shifts
    # y_path is index in 0..nrows-1.
    # shift = index - maxshift
    fshiftv = y_path - maxshift

    # Optional: GreedyQuadPerturbSearchTEST (Fortran) - Skipping for now as it seems optional/diagnostic.

    return fshiftv
