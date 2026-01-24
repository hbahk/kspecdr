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

from kspecdr.tracking import multi_target_tracking
from .wavelets import (
    wavelet_convolution,
    find_resonant_peaks2,
)

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


def windsor_istats(
    ivec: np.ndarray, n: int, cutoff_percent: float
) -> tuple[float, float, bool]:
    """
    Perform Windsor Statistics analysis on an integer vector.
    Returns (mean, sd, check_flag).

    Calculates the mean and standard deviation of the data, but first
    truncates the data to the range [cutoff_percent, 100-cutoff_percent]
    (actually just 100-cutoff_percent logic here to match Fortran description approx).

    Actually, standard Windsor statistics usually involves replacing tails.
    The Fortran code likely does a robust mean/sd calculation assuming X% of data is good.
    Given "assumption that 75% of the data is good" in comments.

    Implementation based on typical Windsorized Mean/SD or Truncated Mean/SD logic.
    Here we implement a simplified version consistent with common robust stats:
    Sort data, take the middle `cutoff_percent` (e.g. 75%), calculate mean/std of that.

    Returns
    -------
    mean : float
    sd : float
    check_flag : bool
        True if quality seems okay (sd < mean + some tolerance?), or just always True unless empty?
        Fortran code says: "Output warning if sanity check fails."
    """
    if n < 1:
        return 0.0, 0.0, False

    # Filter out zeros or bad values if needed? Fortran passes N_ARCLINES found.
    # Assuming IVEC contains counts.

    # Sort the vector
    sorted_vec = np.sort(ivec[:n])

    # Determine indices for truncation
    # e.g. 75% -> we keep the "best" 75%? Or the middle 75%?
    # Usually "assumption 75% good" implies outliers might be 25%.
    # We will just take the interquartile range or similar.
    # Let's keep the middle `cutoff_percent` percent.

    # 1. Selection
    # If 75%, we might discard top 12.5% and bottom 12.5%.
    # But usually for "number of arc lines", low numbers are bad (outliers), high numbers are good/normal.
    # Or high numbers might be noise.
    # Let's stick to standard trimmed mean behavior: trim both ends.

    k = int(n * (100.0 - cutoff_percent) / 200.0)  # Amount to trim from each end
    # e.g. 75% -> trim 12.5% from each end.
    # if n=100, trim 12 from each end. keep 25-100? No.
    # trim = 100 * 25 / 200 = 12.5.

    start_idx = k
    end_idx = n - k

    if start_idx >= end_idx:
        # Fallback to full stats if too few points
        subset = sorted_vec
    else:
        subset = sorted_vec[start_idx:end_idx]

    if len(subset) == 0:
        return 0.0, 0.0, False

    win_mn = float(np.mean(subset))
    win_sd = float(np.std(subset))

    # Sanity check logic
    # "Warning quality of fibre arclines may be compromised"
    # If SD is very high compared to Mean?
    # Or if Mean is too low?
    # Fortran usually does: IF (ABS(WIN_MN - MEDIAN) > ...) or if SD > ...
    # Without the Fortran source for WINDSOR_ISTATS, we assume a basic check.
    # If SD > Mean, it's definitely suspicious for counts.
    chk_flag = True
    if win_mn > 0 and (win_sd / win_mn) > 0.5:
        # If variation is > 50% of mean, that's messy.
        chk_flag = False

    return win_mn, win_sd, chk_flag


def landmark_register(
    spectra: np.ndarray,
    npix: int,
    nfib: int,
    maskv: np.ndarray,
    ref_fib: int,
    scale: float,
    ztol: float,
    diagnostic: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Register and align landmarks.

    Parameters
    ----------
    spectra : np.ndarray
        Input spectra (npix, nfib) - Fortran uses (NPIX, NFIB)
    npix : int
        Number of pixels
    nfib : int
        Number of fibers
    maskv : np.ndarray
        Mask vector (True if masked/bad)
    ref_fib : int
        Reference fiber index
    scale : float
        Wavelet scale
    ztol : float
        Zero tolerance for peak finding
    diagnostic : bool
        Whether to print diagnostic info

    Returns
    -------
    lmr : np.ndarray
        Landmark Register Array (nfib, nlm)
        Note: Fortran defines LMR(NFIB, NPIX) but fills it sparsely?
        The python return should be (nfib, nlm) where nlm is the number of found landmarks.
        Wait, Fortran: REAL, INTENT(OUT) :: LMR(NFIB,NPIX).
        And: LMR(FIBNO,USEIDX)=TRACKA(I,SEQIDX).
        So it returns a 2D array where the 2nd dimension is the landmark index (up to NPIX max).
        We will return (nfib, nlm) sized array.
    nlm : int
        Number of landmarks
    """
    logger.info("Landmark Registration...")

    # 1. Find landmarks in each fibre
    t = np.arange(npix, dtype=float)

    # SEQ_A: Sequence array for MTT.
    # Fortran: SEQ_A(NPIX, NFIB) -> stores peak positions.
    # In Python, we can just use a list of arrays or a large array.
    # The Fortran code fills SEQ_A(cnt, nseq) = peak_pos.
    # And keeps a map SEQMAPV(nseq) = fibno.
    # It skips masked fibers.

    # We will emulate this structure for clarity and MTT input.
    # But our MTT expects `pk_grid[step, peak_idx]`.
    # nsteps (sequences) = number of good fibers.

    # First pass: Count good fibers and collect peaks
    good_fibers = []
    peaks_per_fiber = []  # list of arrays

    logger.info("-> Identifying landmarks within each extracted fibre")

    for fib in range(nfib):
        if maskv[fib]:
            continue

        # Progress logging could be added here

        signal = spectra[:, fib]
        # Handle bad values (NaNs or specific bad values)
        # Assumed handled or simple replacement
        signal = np.nan_to_num(signal)

        # Wavelet convolution
        cwt = wavelet_convolution(signal, t, scale)

        # Determine ZTOL
        _ztol = ztol
        if _ztol <= 0:
            # 10% of max
            mx = np.max(cwt)
            _ztol = 0.1 * mx
            # logger.debug(f"Using auto ztol={_ztol} for fiber {fib}")

        # Find peaks
        pks = find_resonant_peaks2(cwt, t, _ztol)

        # Filter peaks close to bad pixels?
        # "Store all peaks found that are not on or next to a bad pixel"
        # We did nan_to_num, so check original spectra for bad values if needed.
        # For now, assume clean or handled by nan_to_num.

        # Add to list
        peaks_per_fiber.append(pks)
        good_fibers.append(fib)

    nseq = len(good_fibers)
    if nseq == 0:
        logger.warning("No good fibers found for landmark registration.")
        return np.zeros((nfib, 0)), 0

    # Sanity Check: Windsor Stats
    counts = np.array([len(p) for p in peaks_per_fiber])
    win_mn, win_sd, chk_flag = windsor_istats(counts, nseq, 75.0)

    if not chk_flag:
        logger.warning("Warning: quality of fibre arclines may be compromised (high variance in line counts)")
        # Continue anyway as per Fortran? "Output warning...". Yes.

    # Prepare Data for MTT
    # MTT expects `pk_grid` of shape (nsteps, max_ntraces).
    # nsteps = nseq.
    # max_ntraces? We don't strictly know, but it's roughly the number of arc lines.
    # Let's find the max number of peaks found in any fiber.
    max_peaks_found = np.max(counts) if len(counts) > 0 else 0
    # Add some buffer? Fortran uses NPIX size for SEQ_A.
    # Our MTT implementation handles sparse arrays if we pass correct shape.
    # Let's allocate `pk_grid` with `max_peaks_found`.

    pk_grid = np.zeros((nseq, max_peaks_found))
    for i, pks in enumerate(peaks_per_fiber):
        n_p = len(pks)
        if n_p > 0:
            pk_grid[i, :n_p] = np.sort(pks)[:max_peaks_found]  # Ensure sorted and fits

    # Run Multi-Target Tracking
    logger.info("Tracking arc landmarks from fibre to fibre")
    # Parameters for MTT
    # Fortran: CALL MULTI_TARGET_TRACKING(SEQ_A,NPIX,NSEQ,NTRACKS,TRACKA,20.0)
    # MAX_DISPLACEMENT = 20.0 (pixels)?
    max_disp = 20.0
    ntracks, tracka = multi_target_tracking(pk_grid, nseq, max_peaks_found, max_disp)

    logger.info(f"Found {ntracks} Tracks")

    # Filter tracks (FIND TRACKS THAT CAN BE USED TO MODEL THE DISTORTION)
    # "Find a set of tracks that are present in over 50% of all sequences."
    # (Comment says 50%, code says > 75% in the provided snippet: `IF (PERCENT>75.0) THEN`)
    # Wait, the snippet says `IF (PERCENT>75.0) THEN` but commented out `IF (PERCENT>50.0)`.
    # I will use 75% to match the active code in the snippet.

    min_percent = 75.0
    valid_track_indices = []

    # tracka shape is (max_ntraces, nseq) = (max_peaks_found, nseq)
    # Fortran output TRACKA(NPIX, NFIB) -> (TrackID, SequenceID).
    # Wait, Fortran TRACKA indices:
    # `TRACKA(I, SEQIDX)` where I is track index, SEQIDX is sequence index.
    # Our `multi_target_tracking` returns `trace_pts` (ntraces, nsteps).
    # which is (max_ntraces, nseq).
    # So `tracka[i, :]` is the i-th track across all sequences.

    for i in range(ntracks):
        # Count non-zeros
        n_present = np.count_nonzero(tracka[i, :])
        percent = 100.0 * n_present / nseq
        if percent > min_percent:
            valid_track_indices.append(i)

    nuse = len(valid_track_indices)
    if nuse < 3:
        logger.warning("Warning! Unable to trace enough strong arcs fully down the image")

    # Compile LMR
    # LMR(NFIB, NLM)
    # We need to map back from Sequence Index to Fiber Index.
    # `good_fibers[seq_idx]` gives the fiber index.

    lmr = np.zeros((nfib, nuse))
    for use_idx, track_idx in enumerate(valid_track_indices):
        track_data = tracka[track_idx, :]  # Shape (nseq,)
        for seq_idx, pos in enumerate(track_data):
            if pos > 0:
                fib_no = good_fibers[seq_idx]
                lmr[fib_no, use_idx] = pos

    return lmr, nuse


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
    Iterates outwards from ref_fib to propagate calibration on failure.
    """
    rebin_spectra = np.zeros_like(spectra)

    axis1 = np.arange(npix, dtype=float)  # Reference axis

    # Split loop into two legs: Down (Ref -> 0) and Up (Ref+1 -> NFIB)
    # Default Identity coeffs for deg=2 (y=x): [0, 1, 0]
    default_coeffs = np.array([0.0, 1.0, 0.0])

    legs = [
        range(ref_fib, -1, -1),
        range(ref_fib + 1, nfib)
    ]

    for leg in legs:
        last_good_coeffs = default_coeffs.copy()

        for fib in leg:
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

            coeffs = last_good_coeffs
            if len(x_pts) >= 3:
                x_pts = np.array(x_pts)
                y_pts = np.array(y_pts)
                # Fit mapping: ref_pos = f(fib_pos).
                coeffs = robust_polyfit(x_pts / npix, y_pts / npix, 2)
                last_good_coeffs = coeffs
            else:
                logger.warning(f"Synchronise Signals: Fibre {fib} has insufficient landmarks ({len(x_pts)}). Using neighbor coefficients.")

            # axis2 = f(axis1)
            axis2_norm = np.polyval(coeffs, axis1 / npix)
            axis2 = axis2_norm * npix

            isfinite = np.isfinite(spectra[:, fib])
            if np.any(isfinite):
                f_interp = interp1d(
                    axis2[isfinite], spectra[:, fib][isfinite], kind="linear", bounds_error=False, fill_value=0.0
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
    Iterates outwards from ref_fib to propagate calibration on failure.

    cal_axis: Calibration of reference fibre (wavelengths).
    """
    synchcal_axes = np.zeros((nfib, npix + 1))

    # cal_axis has length NPIX+1 (edges)
    axis1 = np.arange(npix + 1, dtype=float)

    # Default Identity coeffs for deg=3 (y=x): [0, 0, 1, 0]
    default_coeffs = np.array([0.0, 0.0, 1.0, 0.0])

    legs = [
        range(ref_fib, -1, -1),
        range(ref_fib + 1, nfib)
    ]

    for leg in legs:
        last_good_coeffs = default_coeffs.copy()

        for fib in leg:
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

            coeffs = last_good_coeffs
            if len(x_pts) >= 3:
                x_pts = np.array(x_pts)
                y_pts = np.array(y_pts)
                # Map this fibre pixels -> ref fibre pixels
                coeffs = robust_polyfit(x_pts / npix, y_pts / npix, 3)  # Cubic
                last_good_coeffs = coeffs
            else:
                logger.warning(f"Synchronise Calibration: Fibre {fib} has insufficient landmarks ({len(x_pts)}). Using neighbor coefficients.")

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
