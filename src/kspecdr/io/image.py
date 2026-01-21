"""
Astropy-based I/O utilities for tramline map processing.

This module provides a simple interface for reading image data, header keywords,
and other metadata from FITS files, replacing the Fortran TDFIO functions.

Usage example:

    from kspecdr.io.image import ImageFile
    with ImageFile('myfile.fits') as im_file:
        img = im_file.read_image_data()
        n_rows, n_cols = img.shape
        var = im_file.read_variance_data()
        fibre_types, nf = im_file.read_fiber_types(1000)
        value, comment = im_file.read_header_keyword('SPECTID')
        code = im_file.get_instrument_code()
"""

import numpy as np
from astropy.io import fits
from typing import Tuple, Optional, Dict, Any
import logging
import sys

logger = logging.getLogger(__name__)


class ImageFile:
    """
    Astropy-based image file handler that replaces Fortran im_id functionality.

    This class provides a simple interface for reading FITS files, similar to
    the Fortran TDFIO functions used in the original 2dfdr code.
    """

    def __init__(self, filename: str, mode: str = "READ"):
        """
        Initialize the image file handler.

        Parameters
        ----------
        filename : str
            Path to the FITS file
        mode : str, optional
            File access mode ('READ', 'UPDATE', 'WRITE'). Default is 'READ'.
        """
        self.filename = filename
        self.mode = mode.upper()
        self.hdul = None
        self._nx = None
        self._ny = None

    def open(self, mode: str = None) -> None:
        """
        Open the FITS file.

        Parameters
        ----------
        mode : str, optional
            File access mode ('READ', 'UPDATE', 'WRITE').
            If None, uses the mode set during initialization.
        """
        if mode is None:
            mode = self.mode

        if mode.upper() == "READ":
            self.hdul = fits.open(self.filename, mode="readonly")
        elif mode.upper() == "UPDATE":
            self.hdul = fits.open(self.filename, mode="update")
        elif mode.upper() == "WRITE":
            self.hdul = fits.open(self.filename, mode="write")
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
        Get the image dimensions (FITS convention).

        Returns
        -------
        tuple
            (nx, ny) dimensions of the image, where nx is width (NAXIS1) and ny is height (NAXIS2).
            Note: This corresponds to a numpy array of shape (ny, nx).
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

    def read_image_data(self) -> np.ndarray:
        """
        Read image data from the primary HDU.

        Returns
        -------
        np.ndarray
            Image data array with shape (rows, cols) [equivalent to (NAXIS2, NAXIS1)]
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        data = self.hdul[0].data
        if data is None:
            raise ValueError("No image data found in primary HDU")

        # # Transpose to match Fortran convention (spectral, spatial)
        # data = data.T

        return data.astype(np.float32)

    def write_image_data(self, data: np.ndarray) -> None:
        """
        Write image data to the primary HDU.

        Parameters
        ----------
        data : np.ndarray
            Image data array with shape (nx, ny)

        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Transpose to match Fortran convention (spectral, spatial)
        data = data.T

        # Write data to primary HDU
        self.hdul[0].data = data

        nx, ny = data.shape

        # Update header with new dimensions
        self.hdul[0].header["NAXIS1"] = nx
        self.hdul[0].header["NAXIS2"] = ny

        # Write to file
        self.hdul.writeto(self.filename, overwrite=True)

    def save_as(self, filename: str) -> None:
        """
        Save the image file to a new filename.

        Parameters
        ----------
        filename : str
            Path to the new FITS file
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Write to new file
        self.hdul.writeto(filename, overwrite=True)

    def save_primary_as(self, filename: str) -> None:
        """
        Save the primary HDU to a new filename.

        Parameters
        ----------
        filename : str
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Write to new file
        self.hdul[self.hdul.index_of("PRIMARY")].writeto(filename, overwrite=True)

    def read_variance_data(self) -> np.ndarray:
        """
        Read variance data from the variance HDU.

        Returns
        -------
        np.ndarray
            Variance data array with shape (rows, cols)
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Look for variance HDU
        var_hdu = None
        for hdu in self.hdul:
            if hdu.name == "VARIANCE" or "VAR" in hdu.name.upper():
                var_hdu = hdu
                break

        if var_hdu is None:
            # Create dummy variance data
            logger.warning("No variance HDU found, creating dummy variance")
            nx, ny = self.get_size()
            return np.ones((ny, nx), dtype=np.float32)

        data = var_hdu.data
        if data is None:
            nx, ny = self.get_size()
            return np.ones((ny, nx), dtype=np.float32)

        # # Transpose to match Fortran convention
        # data = data.T

        return data.astype(np.float32)

    def read_wave_data(self) -> Optional[np.ndarray]:
        """
        Read wavelength data from the WAVELA HDU.

        Returns
        -------
        np.ndarray or None
            Wavelength data array with shape (rows, cols) if found, else None
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Look for WAVELA HDU
        wave_hdu = None
        for hdu in self.hdul:
            if "WAVELA" in hdu.name.upper():
                wave_hdu = hdu
                break

        if wave_hdu is None:
            return None

        data = wave_hdu.data
        if data is None:
            return None

        return data.astype(np.float32)

    def write_wave_data(self, data: np.ndarray) -> None:
        """
        Write wavelength data to the WAVELA HDU.

        Parameters
        ----------
        data : np.ndarray
            Wavelength data array with shape (ny, nx)
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
        wave_hdu = None
        for hdu in self.hdul:
            if "WAVELA" in hdu.name.upper():
                wave_hdu = hdu
                break

        if wave_hdu is None:
            raise ValueError("No WAVELA HDU found")

        wave_hdu.data = data
        ny, nx = data.shape
        wave_hdu.header["NAXIS1"] = ny
        wave_hdu.header["NAXIS2"] = nx

    def write_variance_data(self, data: np.ndarray) -> None:
        """
        Write variance data to the variance HDU.

        Parameters
        ----------
        data : np.ndarray
            Variance data array with shape (ny, nx)
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Transpose to match FITS convention
        data = data.T

        # Find variance HDU
        variance_hdu = None
        for hdu in self.hdul:
            if hdu.name == "VARIANCE":
                variance_hdu = hdu
                break

        if variance_hdu is None:
            raise ValueError("No variance HDU found")

        # Write data to variance HDU
        variance_hdu.data = data

        ny, nx = data.shape

        # Update header with new dimensions
        variance_hdu.header["NAXIS1"] = ny
        variance_hdu.header["NAXIS2"] = nx

    def write_shifts_data(self, data: np.ndarray) -> None:
        """
        Write shifts data to the SHIFTS HDU.

        Parameters
        ----------
        data : np.ndarray
            Shifts data array with shape (ny, nx)
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")
        shifts_hdu = None
        for hdu in self.hdul:
            if "SHIFTS" in hdu.name.upper():
                shifts_hdu = hdu
                
        if shifts_hdu is None:
            # Create new SHIFTS HDU if it doesn't exist
            from astropy.io import fits
            ny, nx = data.shape
            shifts_hdu = fits.ImageHDU(data=data, name="SHIFTS")
            shifts_hdu.header["NAXIS1"] = ny
            shifts_hdu.header["NAXIS2"] = nx
            self.hdul.append(shifts_hdu)
        else:
            # Update existing HDU
            shifts_hdu.data = data
            ny, nx = data.shape
            shifts_hdu.header["NAXIS1"] = ny
            shifts_hdu.header["NAXIS2"] = nx

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

    def get_header_value(self, keyword: str, default: str = None) -> str:
        """
        Get a header keyword value.

        Parameters
        ----------
        keyword : str
            Header keyword name
        default : str, optional
            Default value if keyword not found

        Returns
        -------
        str
            Keyword value or default value
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        header = self.hdul[0].header
        return header.get(keyword, default)

    def read_fiber_types(self, max_nfibres: int) -> Tuple[np.ndarray, int]:
        """
        Read fibre type information.

        Parameters
        ----------
        max_nfibres : int
            Maximum number of fibres

        Returns
        -------
        tuple
            (fiber_types, nf) - fiber type array and number of fibers
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        if self.has_fiber_table():
            fiber_table = self.read_fiber_table()
            fiber_types = fiber_table["TYPE"]
            nf = len(fiber_types)
        else:
            fiber_types = np.full(
                max_nfibres, "N", dtype="U1"
            )  # Default to 'N' (Not used)
            nf = 0

        return fiber_types, nf

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
        instrument = header.get("INSTRUME", "").upper()

        # Map instrument names to codes (based on Fortran constants)
        instrument_codes = {
            "2DF": 1,  # INST_2DF
            "6DF": 2,  # INST_6DF
            "AAOMEGA": 3,  # INST_AAOMEGA_2DF
            "HERMES": 4,  # INST_HERMES
            "SAMI": 5,  # INST_AAOMEGA_SAMI
            "TAIPAN": 6,  # INST_TAIPAN
            "KOALA": 7,  # INST_AAOMEGA_KOALA
            "IFU": 8,  # INST_AAOMEGA_IFU
            "HECTOR": 9,  # INST_SPECTOR_HECTOR
            "AAOMEGA_HECTOR": 10,  # INST_AAOMEGA_HECTOR
            "ISOPLANE": 99,  # INST_ISOPLANE
        }

        for name, code in instrument_codes.items():
            if name in instrument:
                return code

        # Default to generic
        logger.warning(f"Unknown instrument: {instrument}, using generic code")
        return 0  # INST_GENERIC

    def has_variance(self) -> bool:
        """
        Check if the file has a variance HDU.

        Returns
        -------
        bool
            True if variance HDU exists, False otherwise
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Check for variance HDU
        for hdu in self.hdul:
            if hdu.name == "VARIANCE":
                return True
        return False

    def has_fiber_table(self) -> bool:
        """
        Check if the file has a fiber table HDU.

        Returns
        -------
        bool
            True if fiber table HDU exists, False otherwise
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Check for fiber table HDU
        for hdu in self.hdul:
            # if hdu.name in ['FIBRES', 'FIBRES_IFU']:
            if "FIBRES" in hdu.name:
                return True
        return False

    def add_history(self, history: str) -> None:
        """
        Add a history record to the primary HDU.

        Parameters
        ----------
        history : str
            History record to add
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Add history to primary HDU
        self.hdul[0].header["HISTORY"] = history

    def read_fiber_table(self) -> Optional[np.ndarray]:
        """
        Read fiber table data.

        Returns
        -------
        np.ndarray or None
            Fiber table data if present, None otherwise
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Find fiber table HDU
        fiber_hdu = None
        for hdu in self.hdul:
            if "FIBRES" in hdu.name:
                fiber_hdu = hdu
                break

        if fiber_hdu is None:
            return None

        # Return fiber table data
        return fiber_hdu.data

    def write_fiber_table(
        self, fiber_data: np.ndarray, table_name: str = "FIBRES"
    ) -> None:
        """
        Write fiber table data.

        Parameters
        ----------
        fiber_data : np.ndarray
            Fiber table data
        table_name : str, optional
            Name of the fiber table HDU ('FIBRES' or 'FIBRES_IFU')
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Create fiber table HDU
        fiber_hdu = fits.BinTableHDU(fiber_data, name=table_name)
        fiber_hdu.header["EXTNAME"] = table_name

        # Add to HDU list
        self.hdul.append(fiber_hdu)

    def copy_fiber_table_from(self, source_file: "ImageFile") -> None:
        """
        Copy fiber table from another file.

        Parameters
        ----------
        source_file : ImageFile
            Source file containing fiber table
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Read fiber table from source
        fiber_data = source_file.read_fiber_table()
        if fiber_data is None:
            logger.warning("No fiber table found in source file")
            return

        # Determine table name
        table_name = "FIBRES"
        # for hdu in source_file.hdul:
        #     if hdu.name in ['FIBRES', 'FIBRES_IFU']:
        #         table_name = hdu.name
        #         break

        # Write fiber table to current file
        self.write_fiber_table(fiber_data, table_name)
        logger.info(f"Copied fiber table '{table_name}' from source file")

    def get_fiber_table_name(self) -> Optional[str]:
        """
        Get the name of the fiber table HDU.

        Returns
        -------
        str or None
            Name of fiber table HDU if present, None otherwise
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        for hdu in self.hdul:
            if hdu.name in ["FIBRES", "FIBRES_IFU"]:
                return hdu.name
        return None

    def remove_fibers_beyond(self, max_fibers: int) -> None:
        """
        Remove fibers beyond a certain number (for TAIPAN).

        Parameters
        ----------
        max_fibers : int
            Maximum number of fibers to keep
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Find fiber table HDU
        fiber_hdu = None
        for hdu in self.hdul:
            if hdu.name in ["FIBRES", "FIBRES_IFU"]:
                fiber_hdu = hdu
                break

        if fiber_hdu is None:
            logger.warning("No fiber table found")
            return

        # Get number of rows
        nrows = len(fiber_hdu.data)

        if nrows > max_fibers:
            logger.info(f"Removing fibers beyond {max_fibers} (current: {nrows})")

            # Create new fiber table with limited rows
            new_fiber_data = fiber_hdu.data[:max_fibers]

            # Replace the fiber table
            fiber_hdu.data = new_fiber_data

            logger.info(f"Fiber table reduced to {max_fibers} fibers")
        else:
            logger.info(
                f"Fiber table has {nrows} fibers (within limit of {max_fibers})"
            )

    def set_class(self, class_type: str) -> None:
        """
        Set the CLASS keyword in the primary HDU.

        Parameters
        ----------
        class_type : str
            Class type to set
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Set CLASS keyword in primary HDU
        self.hdul[0].header["CLASS"] = class_type

    def delete_keyword(self, keyword: str) -> None:
        """
        Delete a keyword from the primary HDU.

        Parameters
        ----------
        keyword : str
            Keyword to delete
        """
        if self.hdul is None:
            raise RuntimeError("File not opened")

        # Delete keyword from primary HDU
        if keyword in self.hdul[0].header:
            del self.hdul[0].header[keyword]

    def __enter__(self):
        self.open()  # Uses self.mode from initialization
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()