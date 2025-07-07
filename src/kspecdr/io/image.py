"""
Astropy-based I/O utilities for tramline map processing.

This module provides a simple interface for reading image data, header keywords,
and other metadata from FITS files, replacing the Fortran TDFIO functions.

Usage example:

    from kspecdr.io.image import ImageFile
    with ImageFile('myfile.fits') as im_file:
        nx, ny = im_file.get_size()
        img = im_file.read_image_data(nx, ny)
        var = im_file.read_variance_data(nx, ny)
        fibre_types, nf = im_file.read_fibre_types(1000)
        value, comment = im_file.read_header_keyword('SPECTID')
        code = im_file.get_instrument_code()
"""

import numpy as np
from astropy.io import fits
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ImageFile:
    """
    Astropy-based image file handler that replaces Fortran im_id functionality.
    
    This class provides a simple interface for reading FITS files, similar to
    the Fortran TDFIO functions used in the original 2dfdr code.
    """
    
    def __init__(self, filename: str):
        """
        Initialize the image file handler.
        
        Parameters
        ----------
        filename : str
            Path to the FITS file
        """
        self.filename = filename
        self.hdul = None
        self._nx = None
        self._ny = None
        
    def open(self, mode: str = 'READ') -> None:
        """
        Open the FITS file.
        
        Parameters
        ----------
        mode : str, optional
            File access mode ('READ', 'UPDATE', 'WRITE'). Default is 'READ'.
        """
        if mode.upper() == 'READ':
            self.hdul = fits.open(self.filename, mode='readonly')
        elif mode.upper() == 'UPDATE':
            self.hdul = fits.open(self.filename, mode='update')
        elif mode.upper() == 'WRITE':
            self.hdul = fits.open(self.filename, mode='write')
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        # Get image dimensions
        if len(self.hdul) > 0:
            data = self.hdul[0].data
            if data is not None:
                self._ny, self._nx = data.shape
            else:
                self._nx = self._ny = 0
                
        logger.info(f"Opened file: {self.filename} (mode: {mode})")
        
    def close(self) -> None:
        """Close the FITS file."""
        if self.hdul is not None:
            self.hdul.close()
            self.hdul = None
            logger.info(f"Closed file: {self.filename}")
            
    def get_size(self) -> Tuple[int, int]:
        """
        Get the image dimensions.
        
        Returns
        -------
        tuple
            (nx, ny) dimensions of the image
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
            
        if self._nx is None or self._ny is None:
            data = self.hdul[0].data
            if data is not None:
                self._ny, self._nx = data.shape
            else:
                self._nx = self._ny = 0
                
        return self._nx, self._ny
        
    def read_image_data(self, nx: int, ny: int) -> np.ndarray:
        """
        Read image data from the primary HDU.
        
        Parameters
        ----------
        nx : int
            Expected x dimension
        ny : int
            Expected y dimension
            
        Returns
        -------
        np.ndarray
            Image data array with shape (nx, ny)
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
            
        data = self.hdul[0].data
        if data is None:
            raise ValueError("No image data found in primary HDU")
            
        # Transpose to match Fortran convention (spectral, spatial)
        data = data.T
        
        # Check dimensions
        if data.shape != (nx, ny):
            logger.warning(f"Expected shape ({nx}, {ny}), got {data.shape}")
            
        return data.astype(np.float32)
        
    def read_variance_data(self, nx: int, ny: int) -> np.ndarray:
        """
        Read variance data from the variance HDU.
        
        Parameters
        ----------
        nx : int
            Expected x dimension
        ny : int
            Expected y dimension
            
        Returns
        -------
        np.ndarray
            Variance data array with shape (nx, ny)
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
            
        # Look for variance HDU
        var_hdu = None
        for hdu in self.hdul:
            if hdu.name == 'VARIANCE' or 'VAR' in hdu.name.upper():
                var_hdu = hdu
                break
                
        if var_hdu is None:
            # Create dummy variance data
            logger.warning("No variance HDU found, creating dummy variance")
            return np.ones((nx, ny), dtype=np.float32)
            
        data = var_hdu.data
        if data is None:
            return np.ones((nx, ny), dtype=np.float32)
            
        # Transpose to match Fortran convention
        data = data.T
        
        if data.shape != (nx, ny):
            logger.warning(f"Expected variance shape ({nx}, {ny}), got {data.shape}")
            
        return data.astype(np.float32)
        
    def read_header_keyword(self, keyword: str) -> Tuple[str, str]:
        """
        Read a header keyword value and comment.
        
        Parameters
        ----------
        keyword : str
            Header keyword name
            
        Returns
        -------
        tuple
            (value, comment) - keyword value and comment
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
            
        header = self.hdul[0].header
        
        try:
            value = header[keyword]
            comment = header.comments[keyword]
        except KeyError:
            logger.warning(f"Keyword '{keyword}' not found in header")
            value = ""
            comment = ""
            
        return str(value), str(comment)
        
    def read_fibre_types(self, max_nfibres: int) -> Tuple[np.ndarray, int]:
        """
        Read fibre type information.
        
        Parameters
        ----------
        max_nfibres : int
            Maximum number of fibres
            
        Returns
        -------
        tuple
            (fibre_types, nf) - fibre type array and number of fibres
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
            
        # Try to read fibre types from header keywords
        fibre_types = np.full(max_nfibres, 'N', dtype='U1')  # Default to 'N' (Not used)
        
        header = self.hdul[0].header
        nf = 0
        
        # Look for fibre type keywords
        for i in range(1, max_nfibres + 1):
            key = f'FIB{i:03d}TYP'
            if key in header:
                fibre_types[i-1] = header[key]
                nf = max(nf, i)
                
        # If no fibre types found in header, try to estimate from image
        if nf == 0:
            nx, ny = self.get_size()
            # Assume fibres are along the spatial direction
            nf = min(ny, max_nfibres)
            logger.info(f"No fibre types in header, assuming {nf} fibres")
            
        return fibre_types, nf
        
    def get_instrument_code(self) -> int:
        """
        Get the instrument code from header.
        
        Returns
        -------
        int
            Instrument code
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
            
        header = self.hdul[0].header
        
        # Try to get instrument from header
        instrument = header.get('INSTRUME', '').upper()
        
        # Map instrument names to codes (based on Fortran constants)
        instrument_codes = {
            '2DF': 1,      # INST_2DF
            '6DF': 2,      # INST_6DF
            'AAOMEGA': 3,  # INST_AAOMEGA_2DF
            'HERMES': 4,   # INST_HERMES
            'SAMI': 5,     # INST_AAOMEGA_SAMI
            'TAIPAN': 6,   # INST_TAIPAN
            'KOALA': 7,    # INST_AAOMEGA_KOALA
            'IFU': 8,      # INST_AAOMEGA_IFU
            'HECTOR': 9,   # INST_SPECTOR_HECTOR
            'AAOMEGA_HECTOR': 10,  # INST_AAOMEGA_HECTOR
        }
        
        for name, code in instrument_codes.items():
            if name in instrument:
                return code
                
        # Default to generic
        logger.warning(f"Unknown instrument: {instrument}, using generic code")
        return 0  # INST_GENERIC

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 