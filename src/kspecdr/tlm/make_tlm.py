"""
Tramline map generation using astropy-based I/O.

This module provides functions for generating tramline maps from FITS images,
replacing the Fortran TDFIO functions with astropy-based equivalents.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

from kspecdr.io.image import ImageFile

logger = logging.getLogger(__name__)

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
    im_fname = args.get('IMAGE_FILENAME')
    if not im_fname:
        raise ValueError("IMAGE_FILENAME is required")
    tlm_fname = args.get('TLMAP_FILENAME')
    if not tlm_fname:
        tlm_fname = im_fname.replace('.fits', '_tlm.fits')
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
    if instrument_code == INST_2DF:
        make_tlm_2df(im_file, tlm_fname)
    else:
        make_tlm_other(im_file, tlm_fname, instrument_code, args)


def make_tlm_other(im_file: ImageFile, tlm_fname: str, instrument_code: int, args: Dict[str, Any]) -> None:
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
    img_data, var_data, fibre_types, spectid = read_instrument_data(im_file, instrument_code)
    
    # Step 1: Set instrument-specific parameters
    order, pk_search_method, do_distortion, sparse_fibs, experimental, qad_pksearch = set_instrument_specific_params(instrument_code, args)
    
    # Step 2: Convert fibre types to trace status
    fibre_has_trace = convert_fibre_types_to_trace_status(instrument_code, fibre_types, len(fibre_types))
    
    # Step 3: Count fibre types
    n_officially_inuse = count_fibres_with_trace(fibre_has_trace, 'YES')
    n_potentially_able = count_fibres_with_trace(fibre_has_trace, 'MAYBE')
    n_officially_dead = count_fibres_with_trace(fibre_has_trace, 'NO')
    
    logger.info(f"Fibres officially in use: {n_officially_inuse}")
    logger.info(f"Fibres potentially able: {n_potentially_able}")
    logger.info(f"Fibres officially dead: {n_officially_dead}")
    
    # Step 4: Find fibre traces across the image
    traces, sigmap, spat_slice, pk_posn = detect_traces(
        img_data, order, pk_search_method, do_distortion, 
        sparse_fibs, experimental, qad_pksearch
    )
    
    # Step 5: Match located traces to fibre index
    match_vector, modelled_fibre_positions = match_traces_to_fibres(
        instrument_code, traces, fibre_types, pk_posn, args
    )
    
    # Step 6: Convert identified traces to fibre tramline map array
    tramline_map = convert_traces_to_tramline_map(traces, match_vector, len(fibre_types))
    
    # Step 7: Interpolate missing fibre traces
    if instrument_code == INST_TAIPAN:
        interpolate_tramlines_taipan(tramline_map, match_vector, modelled_fibre_positions)
    else:
        interpolate_tramlines(tramline_map, match_vector, get_fibre_separation(instrument_code))
    
    # Step 8: Write tramline data to output file
    write_tramline_data(tlm_fname, tramline_map, instrument_code, args)
    
    # Step 9: Calculate and write wavelength data (if not 2DF)
    if instrument_code != INST_2DF:
        wavelength_data = predict_wavelength(im_file, tramline_map, args)
        write_wavelength_data(tlm_fname, wavelength_data)
    
    logger.info("Tramline map generation completed")


def read_instrument_data(im_file: ImageFile, instrument_code: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
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
        (img_data, var_data, fibre_types, spectid)
    """
    nx, ny = im_file.get_size()
    img_data = im_file.read_image_data(nx, ny)
    var_data = im_file.read_variance_data(nx, ny)
    fibre_types, nf = im_file.read_fibre_types(MAX__NFIBRES)
    spectid = ""
    if instrument_code == INST_HERMES:
        spectid, _ = im_file.read_header_keyword('SPECTID')
        logger.info(f"HERMES SPECTID: {spectid}")
    return img_data, var_data, fibre_types, spectid


def set_instrument_specific_params(instrument_code: int, args: Dict[str, Any]) -> Tuple[int, int, bool, bool, bool, bool]:
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
    sparse_fibs = args.get('SPARSE_FIBS', False)
    experimental = args.get('TLM_FIT_RES', False)
    qad_pksearch = args.get('QAD_PKSEARCH', False)
    
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
    
    return order, pk_search_method, do_distortion, sparse_fibs, experimental, qad_pksearch


def convert_fibre_types_to_trace_status(instrument_code: int, fibre_types: np.ndarray, nf: int) -> np.ndarray:
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
    fibre_has_trace = np.full(nf, 'NO', dtype='U5')
    
    for i in range(nf):
        fib_type = fibre_types[i]
        
        # Map fibre types to trace status
        if fib_type in ['P', 'S']:  # Program, Sky
            fibre_has_trace[i] = 'YES'
        elif fib_type in ['F', 'D']:  # Fiducial, Dead
            fibre_has_trace[i] = 'NO'
        elif fib_type in ['N', 'U']:  # Not used, Unused
            fibre_has_trace[i] = 'MAYBE'
        else:
            fibre_has_trace[i] = 'NO'
    
    return fibre_has_trace


def count_fibres_with_trace(fibre_has_trace: np.ndarray, status: str) -> int:
    """
    Count fibres with specific trace status.
    
    Parameters
    ----------
    fibre_has_trace : np.ndarray
        Array of trace status
    status : str
        Status to count ('YES', 'NO', 'MAYBE')
        
    Returns
    -------
    int
        Count of fibres with specified status
    """
    return np.sum(fibre_has_trace == status)


def detect_traces(img_data: np.ndarray, order: int, pk_search_method: int, 
                  do_distortion: bool, sparse_fibs: bool, experimental: bool, 
                  qad_pksearch: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect traces in image data.
    
    This function replaces the Fortran LOCATE_TRACES call.
    
    Parameters
    ----------
    img_data : np.ndarray
        Image data
    order : int
        Polynomial order for fitting
    pk_search_method : int
        Peak search method
    do_distortion : bool
        Whether to do distortion modelling
    sparse_fibs : bool
        Whether fibres are sparse
    experimental : bool
        Whether to use experimental settings
    qad_pksearch : bool
        Whether to use quick and dirty peak search
        
    Returns
    -------
    tuple
        (traces, sigmap, spat_slice, pk_posn)
    """
    logger.info("Detecting traces in image data")
    
    # Placeholder implementation - this would call the actual trace detection algorithm
    nx, ny = img_data.shape
    
    # For now, create dummy traces
    n_traces = min(ny // 10, 100)  # Rough estimate
    traces = np.zeros((nx, n_traces))
    sigmap = np.ones((nx, n_traces)) * 1.5
    spat_slice = np.mean(img_data, axis=0)
    pk_posn = np.linspace(10, ny-10, n_traces)
    
    logger.info(f"Detected {n_traces} traces")
    
    return traces, sigmap, spat_slice, pk_posn


def match_traces_to_fibres(instrument_code: int, traces: np.ndarray, 
                          fibre_types: np.ndarray, pk_posn: np.ndarray, 
                          args: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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
    
    # Placeholder implementation - this would implement the actual matching logic
    # based on the instrument-specific routines in the Fortran code
    
    logger.info(f"Matched {np.sum(match_vector > 0)} fibres")
    
    return match_vector, modelled_fibre_positions


def convert_traces_to_tramline_map(traces: np.ndarray, match_vector: np.ndarray, nf: int) -> np.ndarray:
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


def interpolate_tramlines(tramline_map: np.ndarray, match_vector: np.ndarray, sep: float) -> None:
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
        tramline_map[:, fibno] = (1.0 - lambda_val) * tramline_map[:, fibno_below] + lambda_val * tramline_map[:, fibno_above]


def interpolate_tramlines_taipan(tramline_map: np.ndarray, match_vector: np.ndarray, 
                                nominal_positions: np.ndarray) -> None:
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
        lambda_val = (nominal_positions[fibno] - nominal_positions[fibno_below]) / (nominal_positions[fibno_above] - nominal_positions[fibno_below])
        tramline_map[:, fibno] = (1.0 - lambda_val) * tramline_map[:, fibno_below] + lambda_val * tramline_map[:, fibno_above]


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


def write_tramline_data(tlm_fname: str, tramline_map: np.ndarray, 
                       instrument_code: int, args: Dict[str, Any]) -> None:
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
    hdu.header['INSTRUME'] = f'INST_{instrument_code}'
    hdu.header['MWIDTH'] = 1.9  # Median spatial FWHM
    hdu.header['PSF_TYPE'] = 'GAUSS'
    
    # Create HDU list
    hdul = fits.HDUList([hdu])
    
    # Write to file
    hdul.writeto(tlm_fname, overwrite=True)
    hdul.close()


def predict_wavelength(im_file: ImageFile, tramline_map: np.ndarray, args: Dict[str, Any]) -> np.ndarray:
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
    # TODO: ... fallback or other instrument logic ...
    wavelength_data = np.zeros((nx, nf))
    for i in range(nx):
        wavelength_data[i, :] = 5000 + i * 0.1
    return wavelength_data


def predict_wavelength_taipan(im_file: ImageFile, nx: int, nf: int) -> np.ndarray:
    """
    Predict wavelength for TAIPAN instrument (Fortran WLA_TAIPAN equivalent).
    Reads LAMBDAC and DISPERS from FITS header and computes wavelength for each pixel/fibre.
    """
    try:
        lambdac_str, _ = im_file.read_header_keyword('LAMBDAC')
        dispers_str, _ = im_file.read_header_keyword('DISPERS')
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
    hdul = fits.open(tlm_fname, mode='update')
    
    # Create wavelength HDU
    hdu = fits.ImageHDU(wavelength_data.T, name='WAVELA')  # Transpose to match FITS convention
    
    # Add to HDU list
    hdul.append(hdu)
    
    # Write changes
    hdul.flush()
    hdul.close()


def make_tlm_2df(im_file: ImageFile, tlm_fname: str) -> None:
    """
    Generate tramline map for 2DF instrument.
    
    Parameters
    ----------
    im_file : ImageFile
        Opened image file handler
    tlm_fname : str
        Output tramline map filename
    """
    logger.info("Generating tramline map for 2DF instrument")
    
    # Placeholder implementation for 2DF-specific tramline map generation
    # This would implement the Fortran MAKE_TLM_2DF functionality
    
    logger.info("2DF tramline map generation completed")


