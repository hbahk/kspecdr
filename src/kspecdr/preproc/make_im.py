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
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from astropy.io import fits
from astropy.table import Table
import warnings

try:
    import astroscrappy

    HAS_ASTROSCRAPPY = True
except ImportError:
    HAS_ASTROSCRAPPY = False
    logger = logging.getLogger(__name__)
    logger.warning("astroscrappy not available. Cosmic ray removal will not work.")

from ..io.image import ImageFile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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

    def process_raw_to_im(
        self,
        raw_filename: str,
        im_filename: Optional[str] = None,
        use_bias: bool = False,
        bias_filename: Optional[str] = None,
        use_dark: bool = False,
        dark_filename: Optional[str] = None,
        use_lflat: bool = False,
        lflat_filename: Optional[str] = None,
        bad_pixel_mask: Optional[str] = None,
        bad_pixel_mask2: Optional[str] = None,
        mark_saturated: bool = True,
        cosmic_ray_method: str = "NONE",
        **kwargs,
    ) -> str:
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
        use_lflat : bool, optional
            Whether to divide by long-slit flat field
        lflat_filename : str, optional
            Path to long-slit flat field file
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

        # Step 1: Create IM file from raw file with proper processing
        logger.info(
            "Creating IM file from raw data with TDFIO_CREATEBYCOPY functionality..."
        )
        self._create_im_from_raw(raw_filename, im_filename)

        # Step 2: Process the IM file
        with ImageFile(im_filename, mode="UPDATE") as im_file:
            # Step 2: Mark bad pixels
            if self.verbose:
                logger.info("Marking bad pixels...")
            self._mark_bad_pixels(
                im_file, mark_saturated, bad_pixel_mask, bad_pixel_mask2
            )

            # Step 3: Process overscan and subtract bias
            if self.verbose:
                logger.info("Processing overscan and bias subtraction...")
            self._debias_image(im_file, use_bias, bias_filename, **kwargs)

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
            if use_lflat and lflat_filename:
                if self.verbose:
                    logger.info("Applying long-slit flat field...")
                self._divide_by_lflat(im_file, lflat_filename)

            # Step 7: Remove cosmic rays
            if cosmic_ray_method != "NONE":
                if self.verbose:
                    logger.info(f"Removing cosmic rays using {cosmic_ray_method}...")
                self._remove_cosmic_rays(im_file, cosmic_ray_method, **kwargs)

            # Step 8: Save the updated IM file
            im_file.save_as(im_filename)

        if self.verbose:
            logger.info(f"Image data frame {im_filename} created.")

        return im_filename

    def _mark_bad_pixels(
        self,
        im_file: ImageFile,
        mark_saturated: bool,
        bad_pixel_mask: Optional[str] = None,
        bad_pixel_mask2: Optional[str] = None,
    ) -> None:
        """
        Mark bad pixels in the image.

        Parameters
        ----------
        im_file : ImageFile
            The image file to process
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
                        image_data[i - 1, j] = np.nan
                    if i < nx - 1:
                        image_data[i + 1, j] = np.nan
                    if j > 0:
                        image_data[i, j - 1] = np.nan
                    if j < ny - 1:
                        image_data[i, j + 1] = np.nan

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
                raise ValueError(
                    f"Bad pixel mask dimensions {mask_data.shape} "
                    f"don't match image dimensions ({ny}, {nx})"
                )

            return mask_data.T  # Transpose to match our (x, y) convention

    def _debias_image(
        self,
        im_file: ImageFile,
        use_bias: bool,
        bias_filename: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Process overscan region and subtract bias.

        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        use_bias : bool
            Whether bias subtraction is being used
        bias_filename : str, optional
            Path to bias frame file
        **kwargs
            Additional keyword arguments for bias processing
        """
        # Read image data
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)

        if use_bias:
            # Subtract bias frame if provided
            if bias_filename is not None:
                # Read bias frame
                with ImageFile(bias_filename, mode="READ") as bias_file:
                    bias_data = bias_file.read_image_data(nx, ny)
                    # Get bias subtraction method from header or use defaults
                    bias_method = bias_file.get_header_value("BIASTYPE", "MEDIAN")
                    if bias_method == "MEDIAN":
                        bias_level = np.nanmedian(bias_data)
                    elif bias_method == "MASTER":
                        bias_level = bias_data
                    else:
                        raise ValueError(
                            f"Unknown bias subtraction method: {bias_method}"
                        )
                    logger.info(f"Bias subtraction method: {bias_method}")
                    image_data -= bias_level
                    bias_record = np.nanmedian(bias_level)
                    logger.info(f"Subtracted bias level: {bias_record:.2f}")
            else:
                # Get overscan region from header or use defaults
                overscan_region = kwargs.get("overscan_region", None)

                if overscan_region is not None:
                    # Process overscan region
                    bias_level = self._calculate_bias_level(image_data, overscan_region)
                    image_data -= bias_level
                    bias_record = bias_level
                    logger.info("Bias subtraction method: OVERSCAN")
                    logger.info(f"Subtracted bias level: {bias_record:.2f}")
                else:
                    logger.warning("No overscan region found in header")

            # Write back the processed image data
            im_file.write_image_data(image_data)
            im_file.add_history(f"Subtracted bias level: {bias_record:.2f}")
        else:
            logger.info("No bias subtraction performed")
            bias_record = 0.0
            im_file.add_history("No bias subtraction performed")

    def _calculate_bias_level(
        self, image_data: np.ndarray, overscan_region: Tuple[int, int, int, int]
    ) -> float:
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

        # TODO: add option for fitting a polynomial to the overscan region
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
        with ImageFile(dark_filename, mode="READ") as dark_file:
            dark_data = dark_file.read_image_data(nx, ny)

            # Get exposure times
            im_exp = im_file.get_header_value("EXPOSED", 1.0)
            dark_exp = dark_file.get_header_value("EXPOSED", 1.0)

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
        im_file.write_variance_data(variance_data.T)

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
            (noise, gain)
        """
        # (noise_array, gain_array) for each amplifier
        # Try to get amplifier-specific noise and gain
        # noise = np.array([im_file.get_header_value('RDNOISE1', 5.0),
        #                  im_file.get_header_value('RDNOISE2', 5.0),
        #                  im_file.get_header_value('RDNOISE3', 5.0),
        #                  im_file.get_header_value('RDNOISE4', 5.0)])

        # gain = np.array([im_file.get_header_value('GAIN1', 1.0),
        #                 im_file.get_header_value('GAIN2', 1.0),
        #                 im_file.get_header_value('GAIN3', 1.0),
        #                 im_file.get_header_value('GAIN4', 1.0)])
        noise = im_file.get_header_value(
            "RO_NOISE", 7.0
        )  # TODO: fix silent replacement

        gain = im_file.get_header_value("RO_GAIN", 2.0)  # TODO: fix silent replacement

        return noise, gain

    def _calculate_variance(
        self, image_data: np.ndarray, noise: np.ndarray, gain: np.ndarray
    ) -> np.ndarray:
        """
        Calculate variance for each pixel.

        Parameters
        ----------
        image_data : np.ndarray
            Image data array
        noise : float
            Readout noise
        gain : float
            Gain

        Returns
        -------
        np.ndarray
            Variance array
        """
        nx, ny = image_data.shape
        # variance = np.zeros_like(image_data)
        variance = noise**2 + np.maximum(image_data, 0) / gain
        # logger.info(f"noise: {noise}, gain: {gain}, zero variance? {np.any(variance == 0)}")

        # # Determine amplifier layout
        # if len(noise) == 1 or np.all(noise == noise[0]):
        #     # Single amplifier
        #     variance = noise[0]**2 + np.maximum(image_data, 0) / gain[0]
        # elif len(noise) == 2:
        #     # Two amplifiers (left/right)
        #     mid_x = nx // 2
        #     # Left half
        #     variance[:mid_x, :] = noise[0]**2 + np.maximum(image_data[:mid_x, :], 0) / gain[0]
        #     # Right half
        #     variance[mid_x:, :] = noise[1]**2 + np.maximum(image_data[mid_x:, :], 0) / gain[1]
        # elif len(noise) == 4:
        #     # Four amplifiers (quadrants)
        #     mid_x, mid_y = nx // 2, ny // 2
        #     # Bottom-left
        #     variance[:mid_x, :mid_y] = noise[0]**2 + np.maximum(image_data[:mid_x, :mid_y], 0) / gain[0]
        #     # Bottom-right
        #     variance[mid_x:, :mid_y] = noise[1]**2 + np.maximum(image_data[mid_x:, :mid_y], 0) / gain[1]
        #     # Top-right
        #     variance[mid_x:, mid_y:] = noise[2]**2 + np.maximum(image_data[mid_x:, mid_y:], 0) / gain[2]
        #     # Top-left
        #     variance[:mid_x, mid_y:] = noise[3]**2 + np.maximum(image_data[:mid_x, mid_y:], 0) / gain[3]

        return variance

    def _divide_by_lflat(self, im_file: ImageFile, lflat_filename: str) -> None:
        """
        Divide image by long-slit flat field.

        Parameters
        ----------
        im_file : ImageFile
            The image file to process
        lflat_filename : str
            Path to long-slit flat field file
        """
        # Read image data and variance
        nx, ny = im_file.get_size()
        image_data = im_file.read_image_data(nx, ny)
        variance_data = im_file.read_variance_data(nx, ny)

        # Read long-slit flat field
        with ImageFile(lflat_filename, mode="READ") as lflat_file:
            lflat_data = lflat_file.read_image_data(nx, ny)
            lflat_variance = lflat_file.read_variance_data(nx, ny)

            # Check flat field class
            lflat_class = lflat_file.get_header_value("CLASS", "")
            if not lflat_class.startswith("LFLATCAL"):
                logger.warning("Flat field file may not have correct class")

            # Perform division with proper error propagation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Avoid division by zero
                lflat_data = np.where(lflat_data > 0, lflat_data, np.nan)

                # Divide image by flat
                image_data /= lflat_data

                # Propagate errors
                variance_data = variance_data / lflat_data**2 + (
                    image_data**2 * lflat_variance / lflat_data**2
                )

            # Write back the processed data
            im_file.write_image_data(image_data)
            im_file.write_variance_data(variance_data)
            im_file.add_history(f"Divided by long-slit flat field {lflat_filename}")

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
        variance_data = im_file.read_variance_data(nx, ny)

        if method == "LACOSMIC":
            # Use LACosmic algorithm
            cleaned_data = self._lacosmic_clean(
                image_data, variance_data, im_file=im_file, **kwargs
            )
        elif method == "BCLEAN":
            # Use BCLEAN algorithm
            cleaned_data = self._bclean_clean(image_data, **kwargs)
        elif method == "PYCOSMIC":
            # Use Python cosmic ray removal
            cleaned_data = self._pycosmic_clean(image_data, **kwargs)
        else:
            logger.warning(f"Unknown cosmic ray removal method: {method}")
            return

        # Write back the cleaned data
        im_file.write_image_data(cleaned_data)

        # Add history record
        im_file.add_history(f"Removed cosmic rays using {method}")

    def _lacosmic_clean(
        self,
        image_data: np.ndarray,
        variance_data: np.ndarray,
        im_file: Optional[ImageFile] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Clean cosmic rays using LACosmic algorithm via astroscrappy.

        Parameters
        ----------
        image_data : np.ndarray
            Image data to clean
        variance_data : np.ndarray
            Variance data to clean
        im_file : ImageFile, optional
            The image file object to read header keywords (for gain/readnoise)
        **kwargs
            Additional parameters for LACosmic:
            - sigclip: sigma clipping threshold (default: 4.5)
            - sigfrac: fraction of sigma clipping (default: 0.3)
            - objlim: object limit (default: 5.0)
            - gain: gain value (default: None, auto-detect from header)
            - readnoise: read noise (default: None, auto-detect from header)
            - satlevel: saturation level (default: None)
            - verbose: verbose output (default: False)

        Returns
        -------
        np.ndarray
            Cleaned image data
        """
        if not HAS_ASTROSCRAPPY:
            raise ImportError(
                "astroscrappy is required for LACosmic cosmic ray removal. "
                "Install it with: pip install astroscrappy"
            )

        # Default parameters for astroscrappy
        # TODO: add gain, readnoise, satlevel
        default_params = {
            "sigclip": 10.0,
            "sigfrac": 0.5,
            "objlim": 5.0,
            "gain": None,
            "readnoise": None,
            # 'satlevel': self.saturation_value,
            "satlevel": np.inf,
            "verbose": False,
        }

        # Update with any provided kwargs
        params = default_params.copy()
        params.update(kwargs)

        # If gain/readnoise not set, try to read from header (RO_GAIN, RO_NOISE)
        if im_file is not None:
            readnoise, gain = self._get_noise_gain_info(im_file)
            if params["gain"] is None:
                if gain is not None:
                    params["gain"] = gain
                    logger.info(f"Using gain from header (RO_GAIN): {gain}")
            if params["readnoise"] is None:
                if readnoise is not None:
                    params["readnoise"] = readnoise
                    logger.info(f"Using readnoise from header (RO_NOISE): {readnoise}")

        logger.info("Running LACosmic cosmic ray removal with astroscrappy")
        logger.info(f"Parameters: {params}")

        try:
            # Run astroscrappy.detect_cosmics
            # This returns (mask, cleaned_data)
            mask, cleaned_data = astroscrappy.detect_cosmics(
                image_data,
                # invar=variance_data,
                sigclip=params["sigclip"],
                sigfrac=params["sigfrac"],
                objlim=params["objlim"],
                gain=params["gain"],
                readnoise=params["readnoise"],
                satlevel=params["satlevel"],
                verbose=params["verbose"],
                sepmed=False,
            )

            # Count cosmic rays detected
            n_cosmic = np.sum(mask)
            logger.info(f"Detected and removed {n_cosmic} cosmic ray pixels")

            return cleaned_data

        except Exception as e:
            logger.error(f"Error in LACosmic cosmic ray removal: {e}")
            raise RuntimeError(f"LACosmic cosmic ray removal failed: {e}")

    def _bclean_clean(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Clean cosmic rays using BCLEAN-like algorithm via astroscrappy.

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

    def _create_im_from_raw(self, raw_filename: str, im_filename: str) -> None:
        """
        Create IM file from raw file with proper processing.

        This method implements the functionality of TDFIO_CREATEBYCOPY,
        including:
        - Raw file to IM file conversion
        - BITPIX conversion (16-bit integer to 32-bit float)
        - Variance HDU creation
        - Instrument-specific processing (6DF, KOALA, TAIPAN)
        - TAIPAN pre-subtraction
        - Fiber table handling

        Parameters
        ----------
        raw_filename : str
            Path to raw input file
        im_filename : str
            Path to output IM file
        """
        logger.info(f"Creating IM file from raw file: {raw_filename} -> {im_filename}")

        # Open raw file and get basic information
        with ImageFile(raw_filename, mode="READ") as raw_file:
            # Get image dimensions
            nx, ny = raw_file.get_size()

            # Read image data (TDFIO_IMAGE_READ handles integer to float conversion)
            image_data = raw_file.read_image_data(nx, ny)

            # Get instrument information
            instrument = raw_file.get_header_value("INSTRUME", "UNKNOWN")
            spectid = raw_file.get_header_value("SPECTID", "")
            class_type = raw_file.get_header_value("CLASS", "")
            bitpix = raw_file.get_header_value("BITPIX", 16)

            logger.info(
                f"Instrument: {instrument}, SPECTID: {spectid}, CLASS: {class_type}, BITPIX: {bitpix}"
            )

            # Check if this is raw data (16-bit integer)
            is_raw_16bit = bitpix == 16

            # Instrument-specific processing
            is_6df = instrument.upper().startswith("6DF") and is_raw_16bit
            is_koala = instrument.upper().startswith("AAOMEGA-KOALA") and is_raw_16bit
            is_taipan = instrument.upper().startswith("TAIPAN") and is_raw_16bit

            # TAIPAN specific checks
            taipan_is_blue = spectid.upper().startswith("BL")
            taipan_is_red = spectid.upper().startswith("RD")
            is_taipan_science = is_taipan and class_type.startswith("MFOBJECT")
            is_taipan_flat = is_taipan and class_type.startswith("MFFFF")

            # Check if variance exists (definitive proof of raw file)
            has_variance = raw_file.has_variance()
            no_variance_in_target = not has_variance

            # TAIPAN pre-subtraction (scattered light removal)
            if is_taipan_science:
                logger.info(
                    "Raw frame is Taipan Science. Looking for pre-subtract image frame."
                )

                # Determine pre-subtract filename based on color
                if taipan_is_red:
                    sub_filename = "RD_SCI_PRESUB.fits"
                elif taipan_is_blue:
                    sub_filename = "BL_SCI_PRESUB.fits"
                else:
                    sub_filename = None

                if sub_filename and Path(sub_filename).exists():
                    logger.info(f"Pre-subtracting data from {sub_filename}")

                    # Read pre-subtract file
                    with ImageFile(sub_filename, mode="READ") as sub_file:
                        sub_data = sub_file.read_image_data(nx, ny)

                        # Calculate scale from exposure time ratios
                        src_time = raw_file.get_header_value("EXPOSED", 1.0)
                        sub_time = sub_file.get_header_value("EXPOSED", 1.0)
                        scale = src_time / sub_time

                        # Subtract scaled pre-subtract data
                        image_data = image_data - scale * sub_data
                        logger.info(f"Pre-subtracted with scale factor: {scale:.3f}")
                else:
                    logger.info(
                        f"Cannot find file {sub_filename}. Not pre-subtracting."
                    )

            # Instrument-specific image transformations
            # TODO: fix these
            # if is_6df:
            #     logger.info("Processing 6DF raw data: transposing and reversing axes")
            #     # Transpose and reverse spectral axis
            #     image_data = np.flipud(image_data.T)
            #     nx, ny = ny, nx  # Update dimensions

            # elif is_koala:
            #     logger.info("Processing KOALA raw data: flipping spatial axis")
            #     # Flip spatial axis (fiber number indexing from top to bottom)
            #     image_data = np.fliplr(image_data)

            # elif is_taipan:
            #     logger.info("Processing TAIPAN raw data: transposing axes")
            #     # Transpose axes
            #     if taipan_is_blue:
            #         image_data = image_data.T
            #     elif taipan_is_red:
            #         image_data = np.flipud(image_data.T)
            #     else:
            #         image_data = image_data.T
            #     nx, ny = ny, nx  # Update dimensions

        # Create IM file with proper structure
        self._create_im_file_structure(
            im_filename,
            image_data,
            nx,
            ny,
            raw_filename,
            instrument,
            class_type,
            no_variance_in_target,
        )

        logger.info(f"Successfully created IM file: {im_filename}")

    def _create_im_file_structure(
        self,
        im_filename: str,
        image_data: np.ndarray,
        nx: int,
        ny: int,
        raw_filename: str,
        instrument: str,
        class_type: str,
        no_variance_in_target: bool,
    ) -> None:
        """
        Create IM file with proper FITS structure.

        Parameters
        ----------
        im_filename : str
            Output IM filename
        image_data : np.ndarray
            Image data
        nx, ny : int
            Image dimensions
        raw_filename : str
            Original raw filename
        instrument : str
            Instrument name
        class_type : str
            Frame class
        no_variance_in_target : bool
            Whether to create variance HDU
        """
        from astropy.io import fits

        # Open source file for header and fiber table
        source_file = ImageFile(raw_filename, mode="READ")
        source_file.open()
        header = source_file.hdul[source_file.hdul.index_of("PRIMARY")].header

        # Create primary HDU with image data
        primary_hdu = fits.PrimaryHDU(image_data, header=header)

        # Set BITPIX to -32 (32-bit float) for IM files
        primary_hdu.header["BITPIX"] = -32

        # Remove BSCALE and BZERO keywords (not used with float data)
        if "BSCALE" in primary_hdu.header:
            del primary_hdu.header["BSCALE"]
        if "BZERO" in primary_hdu.header:
            del primary_hdu.header["BZERO"]

        # Remove AVGVALUE keyword (can cause problems)
        if "AVGVALUE" in primary_hdu.header:
            del primary_hdu.header["AVGVALUE"]

        # Add processing history
        primary_hdu.header["HISTORY"] = f"Created IM file from {raw_filename}"
        primary_hdu.header["HISTORY"] = f"Instrument: {instrument}"
        primary_hdu.header["HISTORY"] = f"Class: {class_type}"

        # Create HDU list
        hdul = fits.HDUList([primary_hdu])

        # Add variance HDU if needed (raw files don't have variance)
        if no_variance_in_target:
            logger.info("Creating variance HDU for raw file")

            # Create variance HDU at correct extension index
            variance_hdu = fits.ImageHDU(name="VARIANCE")
            variance_hdu.header["EXTNAME"] = "VARIANCE"

            # Initialize variance with zeros (will be calculated later)
            variance_data = np.zeros((ny, nx), dtype=np.float32)
            variance_hdu.data = variance_data

            hdul.append(variance_hdu)

        # Set class in header
        primary_hdu.header["CLASS"] = class_type

        # Write the file
        hdul.writeto(im_filename, overwrite=True)
        hdul.close()

        # Copy fiber table if present and appropriate
        if class_type not in ["BIAS", "DARK"]:
            # Open source file to copy fiber table
            if source_file.has_fiber_table():
                logger.info("Copying fiber table from source file")

                # Open destination file for writing
                with ImageFile(im_filename, mode="UPDATE") as dest_file:
                    dest_file.copy_fiber_table_from(source_file)

                    # Handle TAIPAN fiber table (remove fibers beyond 150)
                    if instrument.upper().startswith("TAIPAN"):
                        logger.info(
                            "Processing TAIPAN fiber table (limiting to 150 fibers)"
                        )
                        dest_file.remove_fibers_beyond(150)

            else:
                logger.info("No fiber table found in source file")

        # close source file
        source_file.close()


def make_im(
    raw_filename: str,
    im_filename: Optional[str] = None,
    use_bias: bool = False,
    use_dark: bool = False,
    dark_filename: Optional[str] = None,
    use_lflat: bool = False,
    lflat_filename: Optional[str] = None,
    bad_pixel_mask: Optional[str] = None,
    bad_pixel_mask2: Optional[str] = None,
    mark_saturated: bool = True,
    cosmic_ray_method: str = "NONE",
    verbose: bool = True,
    **kwargs,
) -> str:
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
    use_lflat : bool, optional
        Whether to divide by long-slit flat field
    lflat_filename : str, optional
        Path to long-slit flat field file
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
        use_lflat=use_lflat,
        lflat_filename=lflat_filename,
        bad_pixel_mask=bad_pixel_mask,
        bad_pixel_mask2=bad_pixel_mask2,
        mark_saturated=mark_saturated,
        cosmic_ray_method=cosmic_ray_method,
        **kwargs,
    )
