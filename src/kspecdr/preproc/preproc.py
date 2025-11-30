"""
Preprocessing and Reduction Routines for KSPEC.

This module implements the high-level reduction functions similar to 2dfdr:
- combine_image: Combine multiple images
- reduce_bias: Process bias frames
- reduce_dark: Process dark frames
- reduce_lflat: Process long-slit flat frames
- reduce_fflat: Process fiber flat frames
- reduce_arc: Process arc frames
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple
from astropy.io import fits

from ..io.image import ImageFile
from .make_im import make_im

logger = logging.getLogger(__name__)

def combine_image(
    input_files: List[str],
    output_file: str,
    method: str = 'MEDIAN',
    adjust_levels: bool = True,
    **kwargs
) -> str:
    """
    Combine multiple images into a single image.
    
    Parameters
    ----------
    input_files : List[str]
        List of input IM filenames
    output_file : str
        Output combined filename
    method : str
        Combination method ('MEDIAN' or 'MEAN'). Default is 'MEDIAN'.
    adjust_levels : bool
        Whether to adjust levels (normalize) before combining. Default is True.
    **kwargs
        Additional arguments
        
    Returns
    -------
    str
        Path to the created combined file
    """
    logger.info(f"Combining {len(input_files)} images into {output_file} using {method}")
    
    if not input_files:
        raise ValueError("No input files provided for combination")
        
    if len(input_files) == 1:
        logger.warning("Only one file provided. Copying input to output.")
        import shutil
        shutil.copy2(input_files[0], output_file)
        return output_file

    # Read the first file to get dimensions and header
    with ImageFile(input_files[0]) as first_file:
        nx, ny = first_file.get_size()
        ref_header = first_file.hdul[0].header
        instrument_code = first_file.get_instrument_code()
        
    # Initialize arrays
    n_files = len(input_files)
    stack_data = np.zeros((n_files, nx, ny), dtype=np.float32)
    stack_var = np.zeros((n_files, nx, ny), dtype=np.float32)
    scales = np.ones(n_files)
    
    # Read all files
    for i, fname in enumerate(input_files):
        logger.info(f"Reading {fname} ({i+1}/{n_files})")
        with ImageFile(fname) as im_file:
            data = im_file.read_image_data(nx, ny)
            var = im_file.read_variance_data(nx, ny)
            
            stack_data[i, :, :] = data
            stack_var[i, :, :] = var
            
            # Calculate scaling if requested (using median of central region)
            if adjust_levels:
                # Use central 50% of image for stats
                x1, x2 = nx // 4, 3 * nx // 4
                y1, y2 = ny // 4, 3 * ny // 4
                med_val = np.nanmedian(data[x1:x2, y1:y2])
                if med_val > 0:
                    scales[i] = med_val
                else:
                    scales[i] = 1.0
                    
    # Normalize scales
    if adjust_levels:
        scales /= np.median(scales)
        logger.info(f"Relative flux scalings: {scales}")
        
        # Apply scaling
        for i in range(n_files):
            stack_data[i, :, :] /= scales[i]
            stack_var[i, :, :] /= (scales[i]**2)
            
    # Combine
    if method.upper() == 'MEDIAN':
        combined_data = np.nanmedian(stack_data, axis=0)
        # Approximate variance for median: Var_med approx (pi/2) * Var_mean
        # Var_mean = sum(Var_i) / N^2
        mean_var = np.nansum(stack_var, axis=0) / (n_files**2)
        combined_var = mean_var * (np.pi / 2.0)
        # More robust variance estimate from data scatter could be:
        # combined_var = np.nanvar(stack_data, axis=0) / n_files
        
    elif method.upper() == 'MEAN':
        combined_data = np.nanmean(stack_data, axis=0)
        combined_var = np.nansum(stack_var, axis=0) / (n_files**2)
        
    else:
        raise ValueError(f"Unknown combination method: {method}")
        
    # Create output file
    # We create a new file based on the header of the first file
    hdu = fits.PrimaryHDU(combined_data.T, header=ref_header)
    
    # Update header
    hdu.header['HISTORY'] = f"Combined {n_files} images using {method}"
    if adjust_levels:
        hdu.header['HISTORY'] = "Flux levels adjusted before combination"
    
    # Create variance HDU
    var_hdu = fits.ImageHDU(combined_var.T, name='VARIANCE')
    
    hdul = fits.HDUList([hdu, var_hdu])
    
    # Handle fiber table if present in first file
    with ImageFile(input_files[0]) as first_file:
        if first_file.has_fiber_table():
             fiber_data = first_file.read_fiber_table()
             if fiber_data is not None:
                 table_name = first_file.get_fiber_table_name() or 'FIBRES'
                 fib_hdu = fits.BinTableHDU(fiber_data, name=table_name)
                 hdul.append(fib_hdu)
                 
    hdul.writeto(output_file, overwrite=True)
    logger.info(f"Created combined file: {output_file}")
    
    return output_file

def reduce_bias(
    raw_files: List[str],
    output_file: str = "BIAScombined.fits",
    **kwargs
) -> str:
    """
    Reduce bias frames: Make IM files and combine them.
    
    Parameters
    ----------
    raw_files : List[str]
        List of raw bias filenames
    output_file : str
        Output master bias filename
        
    Returns
    -------
    str
        Path to the master bias file
    """
    logger.info(f"Reducing {len(raw_files)} bias frames")
    
    im_files = []
    for raw_file in raw_files:
        # Create IM file
        im_file = make_im(
            raw_filename=raw_file,
            use_bias=False,  # Bias frames don't get bias subtracted
            use_dark=False,
            mark_saturated=False,
            **kwargs
        )
        im_files.append(im_file)
        
    # Combine bias frames
    # We don't usually scale bias frames (adjust_levels=False)
    combined_file = combine_image(
        im_files, 
        output_file, 
        method='MEDIAN', 
        adjust_levels=False
    )
        
    return combined_file

def reduce_dark(
    raw_files: List[str],
    output_file: str = "DARKcombined.fits",
    bias_file: Optional[str] = None,
    **kwargs
) -> str:
    """
    Reduce dark frames: Make IM files (subtract bias) and combine them.
    
    Parameters
    ----------
    raw_files : List[str]
        List of raw dark filenames
    output_file : str
        Output master dark filename
    bias_file : str, optional
        Master bias file to subtract
        
    Returns
    -------
    str
        Path to the master dark file
    """
    logger.info(f"Reducing {len(raw_files)} dark frames")
    
    im_files = []
    for raw_file in raw_files:
        # Create IM file with bias subtraction if provided
        im_file = make_im(
            raw_filename=raw_file,
            use_bias=(bias_file is not None),
            bias_filename=bias_file,
            use_dark=False,
            **kwargs
        )
        im_files.append(im_file)
        
    # Combine dark frames
    # Dark frames might need scaling if exposure times differ, but usually they are same
    # adjust_levels=False is safer for darks unless we normalize by exposure time
    combined_file = combine_image(
        im_files, 
        output_file, 
        method='MEDIAN', 
        adjust_levels=False
    )
    
    with ImageFile(combined_file, mode='UPDATE') as im:
        im.set_class('DARKRED')
        
    return combined_file

def reduce_lflat(
    raw_files: List[str],
    output_file: str = "LFLATcombined.fits",
    bias_file: Optional[str] = None,
    dark_file: Optional[str] = None,
    **kwargs
) -> str:
    """
    Reduce long-slit flat frames: Make IM files (sub bias/dark) and combine.
    
    Parameters
    ----------
    raw_files : List[str]
        List of raw lflat filenames
    output_file : str
        Output master lflat filename
    bias_file : str, optional
        Master bias file
    dark_file : str, optional
        Master dark file
        
    Returns
    -------
    str
        Path to the master lflat file
    """
    logger.info(f"Reducing {len(raw_files)} LFLAT frames")
    
    im_files = []
    for raw_file in raw_files:
        im_file = make_im(
            raw_filename=raw_file,
            use_bias=(bias_file is not None),
            bias_filename=bias_file,
            use_dark=(dark_file is not None),
            dark_filename=dark_file,
            **kwargs
        )
        im_files.append(im_file)
        
    # Combine LFLATs
    # We might want to normalize levels for flats
    combined_file = combine_image(
        im_files, 
        output_file, 
        method='MEDIAN', 
        adjust_levels=True
    )
    
    # Post-processing: LFLATs are often spatially smoothed
    # TODO: Add spatial smoothing (e.g. median filter 5x5) if needed
    
    with ImageFile(combined_file, mode='UPDATE') as im:
        im.set_class('LFLATCAL')
        
    return combined_file

def reduce_fflat(
    raw_files: List[str],
    output_file: str = "FFLATcombined.fits",
    bias_file: Optional[str] = None,
    dark_file: Optional[str] = None,
    lflat_file: Optional[str] = None,
    **kwargs
) -> str:
    """
    Reduce fiber flat frames: Make IM files, combine, and prepare for extraction.
    
    Note: Full reduction of fiber flats typically involves extraction (Tramline Map)
    and normalization, which may be handled by subsequent steps or modules.
    This function currently performs the image-level preparation and combination.
    
    Parameters
    ----------
    raw_files : List[str]
        List of raw fflat filenames
    output_file : str
        Output combined fflat filename
    bias_file : str, optional
        Master bias file
    dark_file : str, optional
        Master dark file
    lflat_file : str, optional
        Master lflat file
        
    Returns
    -------
    str
        Path to the combined fflat file
    """
    logger.info(f"Reducing {len(raw_files)} FFLAT frames")
    
    im_files = []
    for raw_file in raw_files:
        im_file = make_im(
            raw_filename=raw_file,
            use_bias=(bias_file is not None),
            bias_filename=bias_file,
            use_dark=(dark_file is not None),
            dark_filename=dark_file,
            use_lflat=(lflat_file is not None),
            lflat_filename=lflat_file,
            **kwargs
        )
        im_files.append(im_file)
        
    # Combine FFLATs
    combined_file = combine_image(
        im_files, 
        output_file, 
        method='MEDIAN', 
        adjust_levels=True
    )
    
    with ImageFile(combined_file, mode='UPDATE') as im:
        im.set_class('MFFFF') # Master Fiber Flat Field Frame? Or just FFLAT
        
    return combined_file

def reduce_arc(
    raw_files: List[str],
    output_file: str = "ARCcombined.fits",
    bias_file: Optional[str] = None,
    dark_file: Optional[str] = None,
    lflat_file: Optional[str] = None,
    **kwargs
) -> str:
    """
    Reduce arc frames: Make IM files, combine, and prepare for extraction.
    
    Parameters
    ----------
    raw_files : List[str]
        List of raw arc filenames
    output_file : str
        Output combined arc filename
    bias_file : str, optional
        Master bias file
    dark_file : str, optional
        Master dark file
    lflat_file : str, optional
        Master lflat file
        
    Returns
    -------
    str
        Path to the combined arc file
    """
    logger.info(f"Reducing {len(raw_files)} ARC frames")
    
    im_files = []
    for raw_file in raw_files:
        im_file = make_im(
            raw_filename=raw_file,
            use_bias=(bias_file is not None),
            bias_filename=bias_file,
            use_dark=(dark_file is not None),
            dark_filename=dark_file,
            use_lflat=(lflat_file is not None),
            lflat_filename=lflat_file,
            **kwargs
        )
        im_files.append(im_file)
        
    # Combine Arcs
    # Arcs shouldn't be flux scaled usually as lines might vary? 
    # But often they are stable. Safe to use MEDIAN without adjust_levels for arcs 
    # to preserve relative line intensities if exposure times are same.
    combined_file = combine_image(
        im_files, 
        output_file, 
        method='MEDIAN', 
        adjust_levels=False
    )
    
    with ImageFile(combined_file, mode='UPDATE') as im:
        im.set_class('MFARC')
        
    return combined_file
