"""
Tramline map generation using astropy-based I/O.

This module provides functions for generating tramline maps from FITS images,
replacing the Fortran TDFIO functions with astropy-based equivalents.

TODO: check the usage of the arguments in the function calls.
"""

import numpy as np
import sys
import logging
from typing import Tuple, Optional, Dict, Any
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from kspecdr.io.image import ImageFile
from .match_fibers import taipan_nominal_fibpos, match_fibers_taipan, match_fibers_isoplane

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Instrument codes (matching Fortran constants)
INST_GENERIC = 0
INST_2DF = 1
INST_6DF = 2
INST_AAOMEGA_2DF = 3
INST_HERMES = 4
INST_AAOMEGA_SAMI = 5
INST_TAIPAN = 6
INST_AAOMEGA_KOALA = 7
INST_AAOMEGA_IFU = 8
INST_SPECTOR_HECTOR = 9
INST_AAOMEGA_HECTOR = 10
INST_ISOPLANE = 99

# Maximum number of fibres
MAX__NFIBRES = 1000


def make_tlm(args: Dict[str, Any]) -> None:
    """
    Generate a tramline map from an image file.

    This function replaces the Fortran MAKE_TLM subroutine.

    Parameters
    ----------
    args : dict
        Dictionary containing method arguments including:
        - 'IMAGE_FILENAME': Input image file path
        - 'TLMAP_FILENAME': Output tramline map file path (optional)
    """
    im_fname = args.get("IMAGE_FILENAME")
    if not im_fname:
        raise ValueError("IMAGE_FILENAME is required")
    tlm_fname = args.get("TLMAP_FILENAME")
    if not tlm_fname:
        tlm_fname = im_fname.replace(".fits", "_tlm.fits")
    logger.info(f"Generating tramline map from {im_fname}")
    with ImageFile(im_fname) as im_file:
        make_tlm_from_im(im_file, tlm_fname, args)
        logger.info(f"Generated tramline map: {tlm_fname}")


def make_tlm_from_im(im_file: ImageFile, tlm_fname: str, args: Dict[str, Any]) -> None:
    """
    Generate tramline map from an opened image file.

    This function replaces the Fortran MAKE_TLM_FROM_IM subroutine.

    Parameters
    ----------
    im_file : ImageFile
        Opened image file handler
    tlm_fname : str
        Output tramline map filename
    args : dict
        Method arguments
    """
    instrument_code = im_file.get_instrument_code()
    logger.info(f"Instrument code: {instrument_code}")

    make_tlm_other(im_file, tlm_fname, instrument_code, args)


def make_tlm_other(
    im_file: ImageFile, tlm_fname: str, instrument_code: int, args: Dict[str, Any]
) -> None:
    """
    Generate tramline map for non-2DF instruments.

    This function replaces the Fortran MAKE_TLM_OTHER subroutine.

    Parameters
    ----------
    im_file : ImageFile
        Opened image file handler
    tlm_fname : str
        Output tramline map filename
    instrument_code : int
        Instrument code
    args : dict
        Method arguments
    """
    logger.info("Starting tramline map generation for non-2DF instrument")

    # Step 0: Pre-amble - Read image data and get instrument information
    img_data, var_data, fibre_types = read_instrument_data(im_file, instrument_code)

    # Extract SPECTID from header and add to args for matching
    spectid = im_file.get_header_value('SPECTID', 'RED')
    args['SPECTID'] = spectid

    # Step 1: Set instrument-specific parameters
    order, pk_search_method, do_distortion, sparse_fibs, experimental, qad_pksearch = (
        set_instrument_specific_params(instrument_code, args)
    )

    # Step 2: Convert fibre types to trace status
    fibre_has_trace = convert_fibre_types_to_trace_status(
        instrument_code, fibre_types, len(fibre_types)
    )

    # Step 3: Count fibre types
    n_officially_inuse = np.sum(fibre_has_trace == "YES")
    n_potentially_able = np.sum(fibre_has_trace == "MAYBE")
    n_officially_dead = np.sum(fibre_has_trace == "NO")

    logger.info(f"Fibres officially in use: {n_officially_inuse}")
    logger.info(f"Fibres potentially able: {n_potentially_able}")
    logger.info(f"Fibres officially dead: {n_officially_dead}")

    # Step 4: Find fiber traces across the image
    nx, ny = img_data.shape
    max_ntraces = len(fibre_types)
    nf = len(fibre_types)

    ntraces, traces, spat_slice, pk_posn = detect_traces(
        img_data,
        nx,
        ny,
        max_ntraces,
        nf,
        order,
        sparse_fibs,
        experimental,
        pk_search_method,
        do_distortion,
    )

    logger.info(f"Found {ntraces} traces across the image")

    # Step 5: Match located traces to fibre index
    match_vector, modelled_fibre_positions = match_traces_to_fibres(
        instrument_code, traces, fibre_types, pk_posn, args
    )

    # Step 6: Convert identified traces to fibre tramline map array
    tramline_map = convert_traces_to_tramline_map(
        traces, match_vector, len(fibre_types)
    )

    # Step 7: Interpolate missing fibre traces
    if instrument_code == INST_TAIPAN:
        interpolate_tramlines_taipan(
            tramline_map, match_vector, modelled_fibre_positions
        )
    else:
        interpolate_tramlines(
            tramline_map, match_vector, get_fibre_separation(instrument_code)
        )

    # Step 8: Write tramline data to output file
    write_tramline_data(tlm_fname, tramline_map, instrument_code, args)

    # Step 9: Calculate and write wavelength data (if not 2DF)
    if instrument_code != INST_2DF:
        wavelength_data = predict_wavelength(im_file, tramline_map, args)
        write_wavelength_data(tlm_fname, wavelength_data)

    logger.info("Tramline map generation completed")


def read_instrument_data(
    im_file: ImageFile, instrument_code: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read instrument data from image file.

    Parameters
    ----------
    im_file : ImageFile
        Opened image file handler
    instrument_code : int
        Instrument code

    Returns
    -------
    tuple
        (img_data, var_data, fibre_types)
    """
    nx, ny = im_file.get_size()
    img_data = im_file.read_image_data(nx, ny)
    var_data = im_file.read_variance_data(nx, ny)
    fibre_types, nf = im_file.read_fiber_types(MAX__NFIBRES)
    return img_data, var_data, fibre_types


def set_instrument_specific_params(
    instrument_code: int, args: Dict[str, Any]
) -> Tuple[int, int, bool, bool, bool, bool]:
    """
    Set instrument-specific parameters.

    Parameters
    ----------
    instrument_code : int
        Instrument code
    args : dict
        Method arguments

    Returns
    -------
    tuple
        (order, pk_search_method, do_distortion, sparse_fibs, experimental, qad_pksearch)
    """
    # Get arguments with defaults
    sparse_fibs = args.get("SPARSE_FIBS", False)
    experimental = args.get("TLM_FIT_RES", False)
    qad_pksearch = args.get("QAD_PKSEARCH", False)

    # Set polynomial order based on instrument
    order = 4  # Default
    if instrument_code == INST_6DF:
        order = 2
    elif instrument_code == INST_TAIPAN:
        order = 2
    elif instrument_code == INST_AAOMEGA_IFU:
        order = 2
    elif instrument_code == INST_AAOMEGA_KOALA:
        order = 2
    elif instrument_code == INST_SPECTOR_HECTOR:
        order = 6
    elif instrument_code == INST_AAOMEGA_HECTOR:
        order = 4

    # Set peak search method
    pk_search_method = 0  # Default (emergence watershed)
    if instrument_code == INST_AAOMEGA_KOALA or instrument_code == INST_AAOMEGA_IFU:
        pk_search_method = 1  # Find all local peaks
    elif instrument_code == INST_TAIPAN:
        pk_search_method = 2  # Wavelet convolution
    elif instrument_code == INST_SPECTOR_HECTOR:
        pk_search_method = 0
    elif instrument_code == INST_AAOMEGA_HECTOR:
        pk_search_method = 0

    # Override with argument if specified
    if qad_pksearch:
        pk_search_method = 1
        logger.info("OVERRIDE PEAK SEARCH METHOD TO QAD")

    # Set distortion modelling flag
    do_distortion = True
    if instrument_code == INST_SPECTOR_HECTOR:
        do_distortion = False
    elif instrument_code == INST_AAOMEGA_HECTOR:
        do_distortion = False

    return (
        order,
        pk_search_method,
        do_distortion,
        sparse_fibs,
        experimental,
        qad_pksearch,
    )


def convert_fibre_types_to_trace_status(
    instrument_code: int, fibre_types: np.ndarray, nf: int
) -> np.ndarray:
    """
    Convert fibre types to trace status.

    Parameters
    ----------
    instrument_code : int
        Instrument code
    fibre_types : np.ndarray
        Array of fibre types
    nf : int
        Number of fibres

    Returns
    -------
    np.ndarray
        Array of trace status ('YES', 'NO', 'MAYBE')
    """
    fibre_has_trace = np.full(nf, "NO", dtype="U5")

    for i in range(nf):
        fib_type = fibre_types[i]

        # Map fibre types to trace status
        if fib_type in ["P", "S"]:  # Program, Sky
            fibre_has_trace[i] = "YES"
        elif fib_type in ["F", "D"]:  # Fiducial, Dead
            fibre_has_trace[i] = "NO"
        elif fib_type in ["N", "U"]:  # Not used, Unused
            fibre_has_trace[i] = "MAYBE"
        else:
            fibre_has_trace[i] = "NO"

    return fibre_has_trace


def detect_traces(
    img_data: np.ndarray,
    nx: int,
    ny: int,
    max_ntraces: int,
    nf: int,
    order: int = 4,
    sparse_fibs: bool = False,
    experimental: bool = False,
    pk_search_mthd: int = 0,
    dodist: bool = True,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect fiber traces across an image.

    This function examines IMG_DATA(NX,NY) for identifiable fibre traces and creates
    a traces pathlist array. It returns a representation of a spatial profile slice
    and peak list that can be used for other analysis.

    This function replaces the Fortran LOCATE_TRACES call.

    Parameters
    ----------
    img_data : np.ndarray
        Image data of shape (nx, ny)
    nx, ny : int
        Dimensions of the image
    max_ntraces : int
        Maximum number of traces to return
    nf : int
        Number of fibers in instrument
    order : int, optional
        Order of polynomial fitting (default: 4)
    sparse_fibs : bool, optional
        If there is only a sparse number of fibers (default: False)
    experimental : bool, optional
        If to use experimental restrictions for blurred data (default: False)
    pk_search_mthd : int, optional
        Peak search method: 0=standard, 1=local peaks, 2=wavelet (default: 0)
    dodist : bool, optional
        Whether to do distortion modeling (default: True)

    Returns
    -------
    tuple
        (ntraces, tracea, rep_slice, rep_pkpos)
        ntraces : int
            Number of traces found
        tracea : np.ndarray
            Trace array of shape (nx, max_ntraces)
        rep_slice : np.ndarray
            Representation profile slice of shape (ny,)
        rep_pkpos : np.ndarray
            Representation slice peak list of shape (ny,)
    """

    # Heuristic parameters (from Fortran code)
    STEP = 50  # Step size for column sweep
    HWID = 10  # Half width for averaging around columns
    MAXD = 4.0  # Maximum displacement expected for fiber traces

    # Initialize arrays
    tracea = np.zeros((nx, max_ntraces))
    rep_slice = np.zeros(ny)
    rep_pkpos = np.zeros(ny)

    # Calculate number of steps
    nsteps = (nx - 1) // STEP + 1

    # Arrays to store peak information
    pk_grid = np.zeros((nsteps, max_ntraces))
    trace_pts = np.zeros((max_ntraces, nsteps))

    # Step 1: Sweep the image to find fiber peaks in selected columns
    logger.info("Sweeping image for signs of fibre traces...")

    # Vectorized column processing
    col_indices = np.arange(0, nx, STEP)
    if col_indices[-1] >= nx:
        col_indices = col_indices[:-1]

    for stepno, colno in enumerate(col_indices):
        # Progress feedback
        perc = float(colno) / float(nx) * 100.0
        logger.info(f"Processing column {colno}/{nx} ({perc:.1f}%)")

        # Create a vector slice by averaging around column colno (vectorized)
        col_start = max(0, colno - HWID)
        col_end = min(nx, colno + HWID + 1)

        # Extract column range and average
        col_range = img_data[col_start:col_end, :]
        valid_mask = ~np.isnan(col_range)

        # Compute average along column axis, handling NaN values
        col_data = np.zeros(ny)
        ngood = np.sum(valid_mask, axis=0)
        valid_cols = ngood > 0

        if np.any(valid_cols):
            col_data[valid_cols] = (
                np.nansum(col_range[:, valid_cols], axis=0) / ngood[valid_cols]
            )

        # Locate fiber peaks in this slice
        if pk_search_mthd == 0:
            # Standard peak finding using scipy with adaptive height threshold
            max_val = np.nanmax(col_data)
            if max_val > 0:
                height_threshold = 0.1 * max_val
                peaks, properties = find_peaks(
                    col_data, height=height_threshold, distance=3
                )
                # Select the highest peaks until the max_ntraces is reached
                if len(peaks) > max_ntraces:
                    peak_heights = properties["peak_heights"]
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    peaks = peaks[sorted_indices[:max_ntraces]]
            else:
                peaks = np.array([], dtype=int)

        elif pk_search_mthd == 1:
            # Quick and dirty method - find all local maxima
            peaks, _ = find_peaks(col_data, distance=2)

            # Filter peaks below 10% of maximum
            if len(peaks) > 0:
                max_height = np.max(col_data[peaks])
                mask = col_data[peaks] >= 0.1 * max_height
                peaks = peaks[mask]

                # Select the highest peaks until the max_ntraces is reached
                if len(peaks) > max_ntraces:
                    peak_heights = col_data[peaks]
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    peaks = peaks[sorted_indices[:max_ntraces]]

        elif pk_search_mthd == 2:
            # Wavelet convolution method
            peaks = _wavelet_peak_detection(col_data, scale=2.0, max_peaks=max_ntraces)
            # Convert peak positions to indices
            if len(peaks) > 0:
                peaks = peaks.astype(int)
            else:
                peaks = np.array([], dtype=int)

        else:
            # Default to standard method
            max_val = np.nanmax(col_data)
            if max_val > 0:
                height_threshold = 0.1 * max_val
                peaks, properties = find_peaks(
                    col_data, height=height_threshold, distance=3
                )
                # Select the highest peaks until the max_ntraces is reached
                if len(peaks) > max_ntraces:
                    peak_heights = properties["peak_heights"]
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    peaks = peaks[sorted_indices[:max_ntraces]]
            else:
                peaks = np.array([], dtype=int)

        # Store peak information
        npks = len(peaks)
        if npks > 0:
            p_pks = peaks.astype(float)
            pk_grid[stepno, :npks] = p_pks

        # Store central slice data for representation
        if stepno == nsteps // 2:
            rep_slice = col_data.copy()
            rep_pkpos[:npks] = p_pks

    # Step 2: Link peak locations into fiber traces
    logger.info("Linking trace data to build fibre Tramline Map...")

    # linking algorithm using clustering approach (different from 2dfdr)
    ntraces, trace_pts = _link_peaks_to_traces(pk_grid, nsteps, max_ntraces, MAXD)

    logger.info(f"Found {ntraces} traces across the image")

    # Step 3: Interpolate across linked points of each identified trace
    logger.info("Interpolating trace paths...")

    x_fit = np.arange(1, nx + 1) - 0.5

    for idx in range(ntraces):
        # Get valid points for this trace
        valid_mask = trace_pts[idx, :] > 0
        if not np.any(valid_mask):
            continue

        x_valid = np.arange(1, nx + 1, STEP)[valid_mask] - 0.5
        y_valid = trace_pts[idx, valid_mask]

        if len(x_valid) < 3:
            # Need at least 3 points for polynomial fitting
            continue

        # Fit polynomial to trace points
        try:
            if order > 4:
                # Use higher order polynomial with regularization
                poly_order = min(order, len(x_valid) - 1)
                coeffs = np.polyfit(x_valid, y_valid, poly_order)
            else:
                # Use quadratic fit
                poly_order = min(2, len(x_valid) - 1)
                coeffs = np.polyfit(x_valid, y_valid, poly_order)

            # Evaluate polynomial across full x range
            y_fit = np.polyval(coeffs, x_fit)
            tracea[:, idx] = y_fit

        except (np.RankWarning, ValueError):
            # If fitting fails, use linear interpolation
            tracea[:, idx] = np.interp(x_fit, x_valid, y_valid)

    # Update ntraces to actual number of valid traces
    ntraces = np.sum([np.any(tracea[:, i] != 0) for i in range(max_ntraces)])

    logger.info(f"Final number of traces: {ntraces}")

    return ntraces, tracea, rep_slice, rep_pkpos


def _link_peaks_to_traces(
    pk_grid: np.ndarray, nsteps: int, max_ntraces: int, max_displacement: float
) -> Tuple[int, np.ndarray]:
    """
    Link peak locations into fiber traces using clustering approach.

    This function implements a simplified version of the Fortran PK_GRID2TRACES
    algorithm, using hierarchical clustering instead of Multi-Target Tracking.

    Parameters
    ----------
    pk_grid : np.ndarray
        Peak grid array
    nsteps : int
        Number of steps
    max_ntraces : int
        Maximum number of traces
    max_displacement : float
        Maximum displacement between consecutive peaks

    Returns
    -------
    tuple
        (ntraces, trace_pts)
        ntraces : int
            Number of traces found
        trace_pts : np.ndarray
            Trace points array of shape (max_ntraces, nsteps)
    """

    # Collect all valid peaks with their positions
    peak_positions = []
    peak_steps = []

    for stepno in range(nsteps):
        peaks_in_step = pk_grid[stepno, :]
        valid_peaks = peaks_in_step[peaks_in_step > 0]

        for peak in valid_peaks:
            peak_positions.append(peak)
            peak_steps.append(stepno)

    if len(peak_positions) == 0:
        return 0, np.zeros((max_ntraces, nsteps))

    # Convert to numpy arrays
    peak_positions = np.array(peak_positions)
    peak_steps = np.array(peak_steps)

    # Create feature matrix for clustering
    # Features: [position, step_number] - similar to Fortran's temporal sequence
    features = np.column_stack([peak_positions, peak_steps])

    # Calculate distance matrix
    distances = pdist(features, metric="euclidean")

    # Perform hierarchical clustering (single linkage for continuity)
    linkage_matrix = linkage(distances, method="single")

    # Determine number of clusters (traces) using distance threshold
    cluster_labels = fcluster(linkage_matrix, max_displacement, criterion="distance")
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    n_clusters = min(n_clusters, max_ntraces)

    # Create trace points array
    trace_pts = np.zeros((max_ntraces, nsteps))

    # Step 1: Assign peaks to traces (similar to Fortran's MTT output)
    for i, (pos, step, label) in enumerate(
        zip(peak_positions, peak_steps, cluster_labels)
    ):
        if label <= max_ntraces:
            trace_pts[label - 1, step] = pos

    # Step 2: Filter traces with significant number of points (like Fortran's 50% threshold)
    significant_traces = []
    for trace_idx in range(n_clusters):
        n_points = np.sum(trace_pts[trace_idx, :] > 0)
        if n_points > 0.5 * nsteps:  # Same threshold as Fortran
            significant_traces.append(trace_idx)

    ntraces = len(significant_traces)

    # Step 3: Sort traces by median position (like Fortran's sorting)
    if ntraces > 0:
        trace_medians = []
        for trace_idx in significant_traces:
            valid_points = trace_pts[trace_idx, :]
            valid_points = valid_points[valid_points > 0]
            if len(valid_points) > 0:
                median_pos = np.median(valid_points)
            else:
                median_pos = 0.0
            trace_medians.append(median_pos)

        # Sort by median position (ascending order)
        sorted_indices = np.argsort(trace_medians)
        significant_traces = [significant_traces[i] for i in sorted_indices]

        # Create final trace array with sorted traces
        final_trace_pts = np.zeros((max_ntraces, nsteps))
        for i, trace_idx in enumerate(significant_traces):
            final_trace_pts[i, :] = trace_pts[trace_idx, :]

        return ntraces, final_trace_pts

    return 0, np.zeros((max_ntraces, nsteps))


def match_traces_to_fibres(
    instrument_code: int,
    traces: np.ndarray,
    fibre_types: np.ndarray,
    pk_posn: np.ndarray,
    args: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match detected traces to fibre indices.

    Parameters
    ----------
    instrument_code : int
        Instrument code
    traces : np.ndarray
        Detected traces
    fibre_types : np.ndarray
        Array of fibre types
    pk_posn : np.ndarray
        Peak positions
    args : dict
        Method arguments

    Returns
    -------
    tuple
        (match_vector, modelled_fibre_positions)
    """
    logger.info("Matching traces to fibres")

    nf = len(fibre_types)
    match_vector = np.zeros(nf, dtype=int)
    modelled_fibre_positions = np.zeros(nf)

    if instrument_code == INST_TAIPAN:
        # Get spectid from args or header?
        # make_tlm_other passed args. Usually SPECTID is in image header.
        # But make_tlm_other doesn't pass header values explicitly here except instrument_code.
        # However, match_traces_to_fibres signature has `args`.
        # Fortran: CALL TDFIO_KYWD_READ_CHAR(IM_ID,'SPECTID',SPECTID,CMT,STATUS)
        # We need access to the image file or pass SPECTID.
        # Current signature: (instrument_code, traces, fibre_types, pk_posn, args)
        # We can assume SPECTID is in args if passed, or we need to change signature/read it.
        # `read_instrument_data` gets `im_file`. `make_tlm_other` has `im_file`.
        # `match_traces_to_fibres` is called from `make_tlm_other`.
        # I should assume SPECTID is passed in args or available.
        # Let's assume it's in args['SPECTID'] which might be populated by caller?
        # Or I should add `spectid` to the function signature.
        # Since I can't easily change the call site in `make_tlm_other` without seeing it (it is in this file).
        # Let's check `make_tlm_other`.

        spectid = args.get('SPECTID', 'RED') # Default to RED

        # Get nominal positions
        nf_taipan = len(fibre_types)
        ar_posn = taipan_nominal_fibpos(spectid, nf_taipan)

        # Match
        match_vector, modelled_fibre_positions = match_fibers_taipan(
            nf_taipan, fibre_types, pk_posn, ar_posn
        )

    elif instrument_code == INST_ISOPLANE:
        # Simple 1-to-1 matching
        match_vector, modelled_fibre_positions = match_fibers_isoplane(
            len(fibre_types), pk_posn
        )

    else:
        raise NotImplementedError(
            f"Trace matching for instrument {instrument_code} not implemented"
        )

    return match_vector, modelled_fibre_positions


def convert_traces_to_tramline_map(
    traces: np.ndarray, match_vector: np.ndarray, nf: int
) -> np.ndarray:
    """
    Convert identified traces to fibre tramline map array.

    Parameters
    ----------
    traces : np.ndarray
        Detected traces
    match_vector : np.ndarray
        Vector matching fibre numbers to trace numbers
    nf : int
        Number of fibres

    Returns
    -------
    np.ndarray
        Tramline map array
    """
    nx, n_traces = traces.shape
    tramline_map = np.zeros((nx, nf))

    n_missing = 0
    for fibno in range(nf):
        traceno = match_vector[fibno]
        if traceno == 0:
            n_missing += 1
            continue
        tramline_map[:, fibno] = traces[:, traceno - 1]  # 0-based indexing

    logger.info(f"Converted traces to tramline map ({n_missing} missing fibres)")

    return tramline_map


def interpolate_tramlines(
    tramline_map: np.ndarray, match_vector: np.ndarray, sep: float
) -> None:
    """
    Interpolate missing fibre traces.

    Parameters
    ----------
    tramline_map : np.ndarray
        Tramline map array
    match_vector : np.ndarray
        Vector matching fibre numbers to trace numbers
    sep : float
        Nominal separation between fibres
    """
    logger.info("Interpolating missing fibre traces")

    nx, nf = tramline_map.shape

    # Find first and last matched fibres
    matched_fibres = np.where(match_vector > 0)[0]
    if len(matched_fibres) < 2:
        logger.warning("Too few matched peaks to interpolate with")
        return

    first_matched = matched_fibres[0]
    last_matched = matched_fibres[-1]

    # Extrapolate from bottom end
    for fibno in range(first_matched - 1, -1, -1):
        delta = (first_matched - fibno) * sep
        tramline_map[:, fibno] = tramline_map[:, first_matched] - delta

    # Extrapolate from top end
    for fibno in range(last_matched + 1, nf):
        delta = (fibno - last_matched) * sep
        tramline_map[:, fibno] = tramline_map[:, last_matched] + delta

    # Interpolate for fibres with neighbours on both sides
    for fibno in range(first_matched + 1, last_matched):
        if match_vector[fibno] != 0:
            continue

        # Find nearest matched fibres above and below
        above_fibres = matched_fibres[matched_fibres > fibno]
        below_fibres = matched_fibres[matched_fibres < fibno]

        if len(above_fibres) == 0 or len(below_fibres) == 0:
            continue

        fibno_above = above_fibres[0]
        fibno_below = below_fibres[-1]

        # Linear interpolation
        lambda_val = (fibno - fibno_below) / (fibno_above - fibno_below)
        tramline_map[:, fibno] = (1.0 - lambda_val) * tramline_map[
            :, fibno_below
        ] + lambda_val * tramline_map[:, fibno_above]


def interpolate_tramlines_taipan(
    tramline_map: np.ndarray, match_vector: np.ndarray, nominal_positions: np.ndarray
) -> None:
    """
    Interpolate missing fibre traces for TAIPAN instrument.

    Parameters
    ----------
    tramline_map : np.ndarray
        Tramline map array
    match_vector : np.ndarray
        Vector matching fibre numbers to trace numbers
    nominal_positions : np.ndarray
        Nominal fibre positions
    """
    logger.info("Interpolating missing fibre traces for TAIPAN")

    nx, nf = tramline_map.shape

    # Similar to interpolate_tramlines but using nominal positions
    matched_fibres = np.where(match_vector > 0)[0]
    if len(matched_fibres) < 2:
        logger.warning("Too few matched peaks to interpolate with")
        return

    first_matched = matched_fibres[0]
    last_matched = matched_fibres[-1]

    # Extrapolate from bottom end
    for fibno in range(first_matched - 1, -1, -1):
        delta = nominal_positions[first_matched] - nominal_positions[fibno]
        tramline_map[:, fibno] = tramline_map[:, first_matched] - delta

    # Extrapolate from top end
    for fibno in range(last_matched + 1, nf):
        delta = nominal_positions[fibno] - nominal_positions[last_matched]
        tramline_map[:, fibno] = tramline_map[:, last_matched] + delta

    # Interpolate for fibres with neighbours on both sides
    for fibno in range(first_matched + 1, last_matched):
        if match_vector[fibno] != 0:
            continue

        above_fibres = matched_fibres[matched_fibres > fibno]
        below_fibres = matched_fibres[matched_fibres < fibno]

        if len(above_fibres) == 0 or len(below_fibres) == 0:
            continue

        fibno_above = above_fibres[0]
        fibno_below = below_fibres[-1]

        # Linear interpolation using nominal positions
        lambda_val = (nominal_positions[fibno] - nominal_positions[fibno_below]) / (
            nominal_positions[fibno_above] - nominal_positions[fibno_below]
        )
        tramline_map[:, fibno] = (1.0 - lambda_val) * tramline_map[
            :, fibno_below
        ] + lambda_val * tramline_map[:, fibno_above]


def get_fibre_separation(instrument_code: int) -> float:
    """
    Get nominal fibre separation for instrument.

    Parameters
    ----------
    instrument_code : int
        Instrument code

    Returns
    -------
    float
        Nominal fibre separation in pixels
    """
    # Default separations (these would be instrument-specific)
    separations = {
        INST_2DF: 4.0,
        INST_6DF: 4.0,
        INST_AAOMEGA_2DF: 4.0,
        INST_HERMES: 4.0,
        INST_TAIPAN: 4.0,
    }

    return separations.get(instrument_code, 4.0)


def write_tramline_data(
    tlm_fname: str, tramline_map: np.ndarray, instrument_code: int, args: Dict[str, Any]
) -> None:
    """
    Write tramline data to output file.

    Parameters
    ----------
    tlm_fname : str
        Output filename
    tramline_map : np.ndarray
        Tramline map array
    instrument_code : int
        Instrument code
    args : dict
        Method arguments
    """
    logger.info(f"Writing tramline data to {tlm_fname}")

    # Create FITS file with tramline map
    from astropy.io import fits

    # Create primary HDU with tramline map
    hdu = fits.PrimaryHDU(tramline_map.T)  # Transpose to match FITS convention

    # Add header keywords
    hdu.header["INSTRUME"] = f"INST_{instrument_code}"
    hdu.header["MWIDTH"] = 1.9  # Median spatial FWHM
    hdu.header["PSF_TYPE"] = "GAUSS"

    # Create HDU list
    hdul = fits.HDUList([hdu])

    # Write to file
    hdul.writeto(tlm_fname, overwrite=True)
    hdul.close()


def predict_wavelength(
    im_file: ImageFile, tramline_map: np.ndarray, args: Dict[str, Any]
) -> np.ndarray:
    """
    Predict wavelength for each pixel along each fibre.

    Parameters
    ----------
    im_file : ImageFile
        Image file handler
    tramline_map : np.ndarray
        Tramline map array
    args : dict
        Method arguments

    Returns
    -------
    np.ndarray
        Wavelength array
    """
    logger.info("Predicting wavelength data")
    nx, nf = tramline_map.shape
    instrument_code = im_file.get_instrument_code()
    if instrument_code == INST_TAIPAN:
        return predict_wavelength_taipan(im_file, nx, nf)
    raise NotImplementedError(
        f"Wavelength prediction for instrument code {instrument_code} not yet implemented. "
        "This should implement the PREDICT_WAVELEN functionality from the Fortran code."
    )


def predict_wavelength_taipan(im_file: ImageFile, nx: int, nf: int) -> np.ndarray:
    """
    Predict wavelength for TAIPAN instrument (Fortran WLA_TAIPAN equivalent).
    Reads LAMBDAC and DISPERS from FITS header and computes wavelength for each pixel/fibre.
    """
    try:
        lambdac_str, _ = im_file.read_header_keyword("LAMBDAC")
        dispers_str, _ = im_file.read_header_keyword("DISPERS")
        lambdac = float(lambdac_str)
        dispers = float(dispers_str)
    except Exception as e:
        logger.error(f"Error reading LAMBDAC or DISPERS from header: {e}")
        raise
    midpix = 0.5 * nx
    wavelength_data = np.zeros((nx, nf), dtype=np.float32)
    for pix in range(nx):
        t = float(pix + 1) - 0.5  # Fortran 1-based index
        dist_from_midpix = t - midpix
        lam = dispers * (dist_from_midpix) + lambdac
        # Fortran code multiplies by 0.1 (presumably to convert to nm)
        value = lam * 0.1
        wavelength_data[pix, :] = value
    return wavelength_data


def write_wavelength_data(tlm_fname: str, wavelength_data: np.ndarray) -> None:
    """
    Write wavelength data to tramline map file.

    Parameters
    ----------
    tlm_fname : str
        Tramline map filename
    wavelength_data : np.ndarray
        Wavelength array
    """
    logger.info("Writing wavelength data")

    from astropy.io import fits

    # Open existing file
    hdul = fits.open(tlm_fname, mode="update")

    # Create wavelength HDU
    hdu = fits.ImageHDU(
        wavelength_data.T, name="WAVELA"
    )  # Transpose to match FITS convention

    # Add to HDU list
    hdul.append(hdu)

    # Write changes
    hdul.flush()
    hdul.close()


def _wavelet_convolution(signal: np.ndarray, scale: float) -> np.ndarray:
    """
    Perform wavelet convolution on a signal.
    
    This function implements a simplified version of the Fortran WAVELET_CONVOLUTION
    using a Mexican hat wavelet.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    scale : float
        Wavelet scale parameter
    
    Returns
    -------
    np.ndarray
        Convolved signal
    """
    try:
        import pywt
        
        # Use Mexican hat wavelet (Ricker wavelet in pywt)
        # This is equivalent to the Fortran implementation
        wavelet = 'mexh'  # Mexican hat wavelet
        
        # Perform continuous wavelet transform
        # scales parameter determines the scale of the wavelet
        scales = np.array([scale])
        coef, freqs = pywt.cwt(signal, scales, wavelet)
        
        # Return the real part of the wavelet coefficients
        return np.real(coef[0, :])
        
    except ImportError:
        # Fallback to simple convolution if pywt is not available
        logger.warning("PyWavelets not available, using simple convolution")
        from scipy import signal as scipy_signal
        
        # Create a simple Gaussian kernel as fallback
        kernel_size = int(4 * scale)
        t = np.linspace(-kernel_size, kernel_size, 2 * kernel_size + 1)
        kernel = np.exp(-0.5 * (t / scale) ** 2)
        kernel = kernel / np.sum(kernel)  # Normalize
        
        # Perform convolution
        convolved = scipy_signal.convolve(signal, kernel, mode="same")
        return convolved


def _find_resonant_peaks_ztol(signal: np.ndarray, ztol: float) -> np.ndarray:
    """
    Find resonant peaks in signal above zero tolerance.

    This function implements the Fortran WAVELET_FIND_RES_PEAKS_ZTOL algorithm.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    ztol : float
        Zero tolerance threshold

    Returns
    -------
    np.ndarray
        Indices of resonant peaks
    """
    peaks = []
    n = len(signal)

    # Find regions above zero tolerance and their maxima
    in_positive_range = False
    beg_idx = 0

    for i in range(1, n - 1):
        if not in_positive_range:
            # Check if we are still in sub-zero range
            if signal[i] < ztol:
                continue

            # We are not in a sub-zero range, mark beginning
            in_positive_range = True
            beg_idx = i

        else:
            # Check if we are still in positive range
            if signal[i] >= ztol:
                continue

            # We have reached an end to positive range, find maximum
            in_positive_range = False
            end_idx = i - 1

            # Find maximum between beg_idx and end_idx
            max_idx = beg_idx
            for j in range(beg_idx, end_idx + 1):
                if signal[j] > signal[max_idx]:
                    max_idx = j

            # Add this peak to the list
            peaks.append(max_idx)

    return np.array(peaks, dtype=int)


def _find_zero_crossings(
    signal: np.ndarray, peaks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find left and right zero crossings for each peak.

    This function implements the Fortran WAVELET_FIND_ZERO_CROSSINGS2 algorithm.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    peaks : np.ndarray
        Peak indices

    Returns
    -------
    tuple
        (lhs_zc, rhs_zc) - left and right zero crossing positions
    """
    lhs_zc = []
    rhs_zc = []

    for peak_idx in peaks:
        if signal[peak_idx] <= 0.0:
            continue

        # Find left zero crossing
        zero_lhs = -1.0
        for j in range(peak_idx, -1, -1):
            if signal[j] < 0.0:
                # Linear interpolation to find zero crossing
                j0, j1 = j, j + 1
                if j1 < len(signal):
                    zero_lhs = j0 - (j1 - j0) / (signal[j1] - signal[j0]) * signal[j0]
                break

        # Find right zero crossing
        zero_rhs = -1.0
        for j in range(peak_idx, len(signal)):
            if signal[j] < 0.0:
                # Linear interpolation to find zero crossing
                j0, j1 = j - 1, j
                if j0 >= 0:
                    zero_rhs = j0 - (j1 - j0) / (signal[j1] - signal[j0]) * signal[j0]
                break

        # Only add if both zero crossings were found
        if zero_lhs >= 0 and zero_rhs >= 0:
            lhs_zc.append(zero_lhs)
            rhs_zc.append(zero_rhs)

    return np.array(lhs_zc), np.array(rhs_zc)


def _wavelet_peak_detection_scipy(
    col_data: np.ndarray, widths: list = None, max_peaks: int = None
) -> np.ndarray:
    """
    Detect peaks using scipy's find_peaks_cwt method.

    This is an alternative to the Fortran-based implementation,
    using scipy's optimized wavelet peak detection.

    Parameters
    ----------
    col_data : np.ndarray
        Column data to analyze
    widths : list, optional
        List of widths for wavelet analysis (default: [2, 4, 8])
    max_peaks : int, optional
        Maximum number of peaks to return (default: None, return all)

    Returns
    -------
    np.ndarray
        Peak positions
    """
    from scipy.signal import find_peaks_cwt

    if widths is None:
        widths = [2, 4, 8]

    # Use scipy's find_peaks_cwt
    peaks = find_peaks_cwt(col_data, widths) # default: ricker wavelet

    # If we have more peaks than max_peaks, select the highest ones
    if max_peaks is not None and len(peaks) > max_peaks:
        # Get peak heights at the peak positions
        peak_heights = col_data[peaks.astype(int)]

        # Sort by peak height (descending) and take top max_peaks
        sorted_indices = np.argsort(peak_heights)[::-1]
        peaks = peaks[sorted_indices[:max_peaks]]

    return peaks.astype(float)


def _wavelet_peak_detection(
    col_data: np.ndarray, scale: float = 2.0, max_peaks: int = None
) -> np.ndarray:
    """
    Detect peaks using wavelet convolution method.

    This function implements the complete wavelet-based peak detection
    algorithm from the Fortran code.

    Parameters
    ----------
    col_data : np.ndarray
        Column data to analyze
    scale : float
        Wavelet scale parameter
    max_peaks : int, optional
        Maximum number of peaks to return (default: None, return all)

    Returns
    -------
    np.ndarray
        Peak positions
    """
    # Step 1: Perform wavelet convolution
    cwt = _wavelet_convolution(col_data, scale)

    # Step 2: Find resonant peaks above zero tolerance
    ztol = 0.1 * np.max(cwt)  # 10% of maximum positive value
    resonant_peaks = _find_resonant_peaks_ztol(cwt, ztol)

    # Step 3: Find zero crossings for each peak
    lhs_zc, rhs_zc = _find_zero_crossings(cwt, resonant_peaks)

    # Step 4: Calculate peak positions as midpoints of zero crossings
    if len(lhs_zc) > 0 and len(rhs_zc) > 0:
        peak_positions = 0.5 * (lhs_zc + rhs_zc)

        # If we have more peaks than max_peaks, select the highest ones
        if max_peaks is not None and len(peak_positions) > max_peaks:
            # Get peak heights at the peak positions
            peak_heights = col_data[peak_positions.astype(int)]

            # Sort by peak height (descending) and take top max_peaks
            sorted_indices = np.argsort(peak_heights)[::-1]
            peak_positions = peak_positions[sorted_indices[:max_peaks]]

        return peak_positions.astype(float)
    else:
        return np.array([], dtype=float)
