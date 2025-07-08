"""
Make IM (Image) Module

This module handles the preprocessing of raw astronomical data to create IM files.
It performs the following steps:
1. Copy raw file to IM file
2. Mark bad pixels
3. Subtract bias and process overscan
4. Subtract dark frame (optional)
5. Create and initialize variance HDU
6. Apply long-slit flat field (optional)
7. Remove cosmic rays (optional)

Based on the Fortran MAKE_IM routine from 2dfdr.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from astropy.io import fits
from astropy.table import Table
import warnings

from ..io.image import ImageFile

logger = logging.getLogger(__name__)

class MakeIM:
    """
    Handles preprocessing of raw astronomical data to create IM files.
    
    This class implements the equivalent of the Fortran MAKE_IM routine,
    performing all necessary preprocessing steps to convert raw instrument
    data into calibrated image files with proper variance estimates.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the MakeIM processor.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose output during processing
        """
        self.verbose = verbose
        self.saturation_value = 65535.0  # 16-bit saturation value
        self.saturation_tolerance = 0.99 * self.saturation_value
        
    def process_raw_to_im(self, 
                          raw_filename: str,
                          im_filename: Optional[str] = None,
                          use_bias: bool = False,
                          use_dark: bool = False,
                          dark_filename: Optional[str] = None,
                          use_flat: bool = False,
                          flat_filename: Optional[str] = None,
                          bad_pixel_mask: Optional[str] = None,
                          bad_pixel_mask2: Optional[str] = None,
                          mark_saturated: bool = True,
                          cosmic_ray_method: str = 'NONE',
                          **kwargs) -> str:
        """
        Process raw file to create IM file with all preprocessing steps.
        
        Parameters
        ----------
        raw_filename : str
            Path to the raw input file
        im_filename : str, optional
            Path for the output IM file. If None, will be derived from raw_filename
        use_bias : bool, optional
            Whether to subtract bias frame
        use_dark : bool, optional
            Whether to subtract dark frame
        dark_filename : str, optional
            Path to dark frame file
        use_flat : bool, optional
            Whether to divide by long-slit flat field
        flat_filename : str, optional
            Path to flat field file
        bad_pixel_mask : str, optional
            Path to bad pixel mask file
        bad_pixel_mask2 : str, optional
            Path to second bad pixel mask file (e.g., cosmic ray mask)
        mark_saturated : bool, optional
            Whether to mark saturated pixels as bad
        cosmic_ray_method : str, optional
            Method for cosmic ray removal ('NONE', 'LACOSMIC', 'BCLEAN', 'PYCOSMIC')
        **kwargs
            Additional keyword arguments
            
        Returns
        -------
        str
            Path to the created IM file
        """
        if self.verbose:
            logger.info("=" * 50)
            logger.info("Preprocessing image data contained in RAW frame")
            logger.info("=" * 50)
            logger.info(f"RAW file = {raw_filename}")
        
        # Determine output filename
        if im_filename is None:
            raw_path = Path(raw_filename)
            im_filename = str(raw_path.with_name(f"{raw_path.stem}_im.fits"))
        
        # Step 1: Copy raw file to IM file
        logger.info("Creating IM file from raw data...")
        with ImageFile(raw_filename) as raw_file:
            # Create IM file by copying raw file
            raw_file.save_as(im_filename)
        
        # Step 2: Process the IM file
        with ImageFile(im_filename, mode='update') as im_file:
            # Step 2: Mark bad pixels
            if self.verbose:
                logger.info("Marking bad pixels...")
            self._mark_bad_pixels(im_file, use_bias, mark_saturated, 
                                 bad_pixel_mask, bad_pixel_mask2)
            
            # Step 3: Process overscan and subtract bias
            if self.verbose:
                logger.info("Processing overscan and bias subtraction...")
            self._debiase_image(im_file, use_bias, **kwargs)
            
            # Step 4: Subtract dark frame
            if use_dark and dark_filename:
                if self.verbose:
                    logger.info("Subtracting dark frame...")
                self._subtract_dark(im_file, dark_filename)
            
            # Step 5: Create and initialize variance HDU
            if self.verbose:
                logger.info("Creating variance HDU...")
            self._add_variance(im_file)
            
            # Step 6: Apply long-slit flat field
            if use_flat and flat_filename:
                if self.verbose:
                    logger.info("Applying long-slit flat field...")
                self._divide_by_flat(im_file, flat_filename)
            
            # Step 7: Remove cosmic rays
            if cosmic_ray_method != 'NONE':
                if self.verbose:
                    logger.info(f"Removing cosmic rays using {cosmic_ray_method}...")
                self._remove_cosmic_rays(im_file, cosmic_ray_method, **kwargs)
        
        if self.verbose:
            logger.info(f"Image data frame {im_filename} created.")
        
        return im_filename
    
    def _mark_bad_pixels(self, 
                         im_file: ImageFile,
                         use_bias: bool,
                         mark_saturated: bool,
                         bad_pixel_mask: Optional[str] = None,
                         bad_pixel_mask2: Optional[str] = None) -> None:
        """
        Mark bad pixels in the image.
        
        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        use_bias : bool
            Whether bias subtraction is being used
        mark_saturated : bool
            Whether to mark saturated pixels as bad
        bad_pixel_mask : str, optional
            Path to bad pixel mask file
        bad_pixel_mask2 : str, optional
            Path to second bad pixel mask file
        """
        # Read image data
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)
        
        # Apply first bad pixel mask if provided
        if bad_pixel_mask and Path(bad_pixel_mask).exists():
            logger.info(f"Using bad pixel mask: {bad_pixel_mask}")
            bad_mask = self._load_bad_pixel_mask(bad_pixel_mask, nx, ny)
            image_data[bad_mask > 0.99] = np.nan
        
        # Apply second bad pixel mask if provided
        if bad_pixel_mask2 and Path(bad_pixel_mask2).exists():
            logger.info(f"Using second bad pixel mask: {bad_pixel_mask2}")
            bad_mask2 = self._load_bad_pixel_mask(bad_pixel_mask2, nx, ny)
            image_data[bad_mask2 > 0.99] = np.nan
        
        # Mark saturated pixels if requested
        if mark_saturated:
            saturated_pixels = image_data >= self.saturation_tolerance
            if np.any(saturated_pixels):
                percent_saturated = np.sum(saturated_pixels) * 100.0 / (nx * ny)
                logger.info(f"{percent_saturated:.2f}% of pixels were saturated")
                
                if percent_saturated > 1.0:
                    logger.warning("Warning: Higher than expected levels of saturation")
                    logger.warning("This may cause serious reduction problems!")
                
                # Mark saturated pixels and neighbors as bad
                image_data[saturated_pixels] = np.nan
                
                # Mark neighboring pixels
                for i, j in zip(*np.where(saturated_pixels)):
                    if i > 0:
                        image_data[i-1, j] = np.nan
                    if i < nx - 1:
                        image_data[i+1, j] = np.nan
                    if j > 0:
                        image_data[i, j-1] = np.nan
                    if j < ny - 1:
                        image_data[i, j+1] = np.nan
        
        # Write back the modified image data
        im_file.write_image_data(image_data)
    
    def _load_bad_pixel_mask(self, mask_filename: str, nx: int, ny: int) -> np.ndarray:
        """
        Load bad pixel mask from file.
        
        Parameters
        ----------
        mask_filename : str
            Path to the bad pixel mask file
        nx, ny : int
            Expected dimensions of the mask
            
        Returns
        -------
        np.ndarray
            Bad pixel mask array
        """
        with fits.open(mask_filename) as hdul:
            mask_data = hdul[0].data
            
            if mask_data.shape != (ny, nx):  # FITS is (y, x) order
                raise ValueError(f"Bad pixel mask dimensions {mask_data.shape} "
                               f"don't match image dimensions ({ny}, {nx})")
            
            return mask_data.T  # Transpose to match our (x, y) convention
    
    def _debiase_image(self, im_file: ImageFile, use_bias: bool, **kwargs) -> None:
        """
        Process overscan region and subtract bias.
        
        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        use_bias : bool
            Whether bias subtraction is being used
        **kwargs
            Additional keyword arguments for bias processing
        """
        # Read image data
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)
        
        # Get overscan region from header or use defaults
        overscan_region = kwargs.get('overscan_region', None)
        
        if overscan_region is not None:
            # Process overscan region
            bias_level = self._calculate_bias_level(image_data, overscan_region)
            image_data -= bias_level
            logger.info(f"Subtracted bias level: {bias_level:.2f}")
        
        # Write back the processed image data
        im_file.write_image_data(image_data)
    
    def _calculate_bias_level(self, image_data: np.ndarray, 
                             overscan_region: Tuple[int, int, int, int]) -> float:
        """
        Calculate bias level from overscan region.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data array
        overscan_region : tuple
            Overscan region as (x1, x2, y1, y2)
            
        Returns
        -------
        float
            Calculated bias level
        """
        x1, x2, y1, y2 = overscan_region
        overscan_data = image_data[x1:x2, y1:y2]
        
        # Use median to avoid outliers
        bias_level = np.nanmedian(overscan_data)
        return bias_level
    
    def _subtract_dark(self, im_file: ImageFile, dark_filename: str) -> None:
        """
        Subtract dark frame from image.
        
        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        dark_filename : str
            Path to dark frame file
        """
        # Read image data
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)
        
        # Read dark frame
        with ImageFile(dark_filename) as dark_file:
            dark_data = dark_file.read_image_data(nx, ny)
            
            # Get exposure times
            im_exp = im_file.get_header_value('EXPOSED', 1.0)
            dark_exp = dark_file.get_header_value('EXPOSED', 1.0)
            
            logger.info(f"Object exposure time: {im_exp}")
            logger.info(f"Dark exposure time: {dark_exp}")
            
            # Scale dark frame
            if dark_exp > 0:
                scale = im_exp / dark_exp
                logger.info(f"Dark scaled by: {scale}")
                dark_data *= scale
            else:
                logger.warning("ERROR: No scaling of dark")
            
            # Subtract dark
            image_data -= dark_data
            
            # Write back the processed image data
            im_file.write_image_data(image_data)
            
            # Add history record
            im_file.add_history(f"Subtracted dark frame {dark_filename}")
    
    def _add_variance(self, im_file: ImageFile) -> None:
        """
        Create and initialize variance HDU.
        
        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        """
        # Read image data
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)
        
        # Get noise and gain information from header
        noise, gain = self._get_noise_gain_info(im_file)
        
        # Calculate variance
        variance_data = self._calculate_variance(image_data, noise, gain)
        
        # Write variance data
        im_file.write_variance_data(variance_data)
        
        logger.info("Variance HDU created and initialized")
    
    def _get_noise_gain_info(self, im_file: ImageFile) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get noise and gain information from FITS header.
        
        Parameters
        ----------
        im_file : ImageFile
            The image file
            
        Returns
        -------
        tuple
            (noise_array, gain_array) for each amplifier
        """
        # Try to get amplifier-specific noise and gain
        noise = np.array([im_file.get_header_value('RDNOISE1', 5.0),
                         im_file.get_header_value('RDNOISE2', 5.0),
                         im_file.get_header_value('RDNOISE3', 5.0),
                         im_file.get_header_value('RDNOISE4', 5.0)])
        
        gain = np.array([im_file.get_header_value('GAIN1', 1.0),
                        im_file.get_header_value('GAIN2', 1.0),
                        im_file.get_header_value('GAIN3', 1.0),
                        im_file.get_header_value('GAIN4', 1.0)])
        
        return noise, gain
    
    def _calculate_variance(self, image_data: np.ndarray, 
                          noise: np.ndarray, gain: np.ndarray) -> np.ndarray:
        """
        Calculate variance for each pixel.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data array
        noise : np.ndarray
            Readout noise for each amplifier
        gain : np.ndarray
            Gain for each amplifier
            
        Returns
        -------
        np.ndarray
            Variance array
        """
        nx, ny = image_data.shape
        variance = np.zeros_like(image_data)
        
        # Determine amplifier layout
        if len(noise) == 1 or np.all(noise == noise[0]):
            # Single amplifier
            variance = noise[0]**2 + np.maximum(image_data, 0) / gain[0]
        elif len(noise) == 2:
            # Two amplifiers (left/right)
            mid_x = nx // 2
            # Left half
            variance[:mid_x, :] = noise[0]**2 + np.maximum(image_data[:mid_x, :], 0) / gain[0]
            # Right half
            variance[mid_x:, :] = noise[1]**2 + np.maximum(image_data[mid_x:, :], 0) / gain[1]
        elif len(noise) == 4:
            # Four amplifiers (quadrants)
            mid_x, mid_y = nx // 2, ny // 2
            # Bottom-left
            variance[:mid_x, :mid_y] = noise[0]**2 + np.maximum(image_data[:mid_x, :mid_y], 0) / gain[0]
            # Bottom-right
            variance[mid_x:, :mid_y] = noise[1]**2 + np.maximum(image_data[mid_x:, :mid_y], 0) / gain[1]
            # Top-right
            variance[mid_x:, mid_y:] = noise[2]**2 + np.maximum(image_data[mid_x:, mid_y:], 0) / gain[2]
            # Top-left
            variance[:mid_x, mid_y:] = noise[3]**2 + np.maximum(image_data[:mid_x, mid_y:], 0) / gain[3]
        
        return variance
    
    def _divide_by_flat(self, im_file: ImageFile, flat_filename: str) -> None:
        """
        Divide image by long-slit flat field.
        
        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        flat_filename : str
            Path to flat field file
        """
        # Read image data and variance
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)
        variance_data = im_file.read_variance_data(nx, ny)
        
        # Read flat field
        with ImageFile(flat_filename) as flat_file:
            flat_data = flat_file.read_image_data(nx, ny)
            flat_variance = flat_file.read_variance_data(nx, ny)
            
            # Check flat field class
            flat_class = flat_file.get_header_value('CLASS', '')
            if not flat_class.startswith('LFLATCAL'):
                logger.warning("Flat field file may not have correct class")
            
            # Perform division with proper error propagation
            # For division: var_result = (image/flat)^2 * (var_image/image^2 + var_flat/flat^2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Avoid division by zero
                flat_data = np.where(flat_data > 0, flat_data, 1.0)
                
                # Divide image by flat
                image_data /= flat_data
                
                # Propagate errors
                variance_data = (image_data**2) * (
                    variance_data / (image_data * flat_data)**2 + 
                    flat_variance / flat_data**2
                )
            
            # Write back the processed data
            im_file.write_image_data(image_data)
            im_file.write_variance_data(variance_data)
            
            # Add history record
            im_file.add_history(f"Divided by long-slit flat field {flat_filename}")
    
    def _remove_cosmic_rays(self, im_file: ImageFile, method: str, **kwargs) -> None:
        """
        Remove cosmic rays from image.
        
        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        method : str
            Method for cosmic ray removal
        **kwargs
            Additional keyword arguments for cosmic ray removal
        """
        # Read image data
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)
        
        if method == 'LACOSMIC':
            # Use LACosmic algorithm
            cleaned_data = self._lacosmic_clean(image_data, **kwargs)
        elif method == 'BCLEAN':
            # Use BCLEAN algorithm
            cleaned_data = self._bclean_clean(image_data, **kwargs)
        elif method == 'PYCOSMIC':
            # Use Python cosmic ray removal
            cleaned_data = self._pycosmic_clean(image_data, **kwargs)
        else:
            logger.warning(f"Unknown cosmic ray removal method: {method}")
            return
        
        # Write back the cleaned data
        im_file.write_image_data(cleaned_data)
        
        # Add history record
        im_file.add_history(f"Removed cosmic rays using {method}")
    
    def _lacosmic_clean(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Clean cosmic rays using LACosmic algorithm.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data to clean
        **kwargs
            Additional parameters for LACosmic
            
        Returns
        -------
        np.ndarray
            Cleaned image data
        """
        raise NotImplementedError(
            "LACosmic cosmic ray removal not yet implemented. "
            "This should implement the LACosmic algorithm for cosmic ray detection and removal."
        )
    
    def _bclean_clean(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Clean cosmic rays using BCLEAN algorithm.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data to clean
        **kwargs
            Additional parameters for BCLEAN
            
        Returns
        -------
        np.ndarray
            Cleaned image data
        """
        raise NotImplementedError(
            "BCLEAN cosmic ray removal not yet implemented. "
            "This should implement the BCLEAN algorithm for cosmic ray detection and removal."
        )
    
    def _pycosmic_clean(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Clean cosmic rays using Python implementation.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data to clean
        **kwargs
            Additional parameters for Python cosmic ray removal
            
        Returns
        -------
        np.ndarray
            Cleaned image data
        """
        raise NotImplementedError(
            "Python cosmic ray removal not yet implemented. "
            "This should implement a Python-based algorithm for cosmic ray detection and removal."
        )


def make_im(raw_filename: str,
            im_filename: Optional[str] = None,
            use_bias: bool = False,
            use_dark: bool = False,
            dark_filename: Optional[str] = None,
            use_flat: bool = False,
            flat_filename: Optional[str] = None,
            bad_pixel_mask: Optional[str] = None,
            bad_pixel_mask2: Optional[str] = None,
            mark_saturated: bool = True,
            cosmic_ray_method: str = 'NONE',
            verbose: bool = True,
            **kwargs) -> str:
    """
    Convenience function to process raw file to IM file.
    
    Parameters
    ----------
    raw_filename : str
        Path to the raw input file
    im_filename : str, optional
        Path for the output IM file. If None, will be derived from raw_filename
    use_bias : bool, optional
        Whether to subtract bias frame
    use_dark : bool, optional
        Whether to subtract dark frame
    dark_filename : str, optional
        Path to dark frame file
    use_flat : bool, optional
        Whether to divide by long-slit flat field
    flat_filename : str, optional
        Path to flat field file
    bad_pixel_mask : str, optional
        Path to bad pixel mask file
    bad_pixel_mask2 : str, optional
        Path to second bad pixel mask file
    mark_saturated : bool, optional
        Whether to mark saturated pixels as bad
    cosmic_ray_method : str, optional
        Method for cosmic ray removal
    verbose : bool, optional
        Whether to print verbose output
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    str
        Path to the created IM file
    """
    processor = MakeIM(verbose=verbose)
    return processor.process_raw_to_im(
        raw_filename=raw_filename,
        im_filename=im_filename,
        use_bias=use_bias,
        use_dark=use_dark,
        dark_filename=dark_filename,
        use_flat=use_flat,
        flat_filename=flat_filename,
        bad_pixel_mask=bad_pixel_mask,
        bad_pixel_mask2=bad_pixel_mask2,
        mark_saturated=mark_saturated,
        cosmic_ray_method=cosmic_ray_method,
        **kwargs
    ) 