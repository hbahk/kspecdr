import numpy as np
from scipy.interpolate import interp1d
import logging
from ..io.image import ImageFile

logger = logging.getLogger(__name__)

def scrunch_open_arc(args):
    """
    Returns the filename of the arc file from args.
    Corresponds to SCRUNCH_OPEN_ARC in Fortran.

    Parameters
    ----------
    args : dict
        Arguments dictionary containing 'WAVEL_FILENAME'.

    Returns
    -------
    str or None
        The filename of the arc file, or None if not found/empty.
    """
    fname = args.get('WAVEL_FILENAME')
    if not fname:
        # Check TDFIO_NOID logic: if empty, return None
        return None
    return fname

def rebin_data(spectra, variance, input_wave, output_wave):
    """
    Rebins spectra and variance from input_wave to output_wave using linear interpolation.

    Parameters
    ----------
    spectra : np.ndarray
        Input spectra (NSPEC, NFIB)
    variance : np.ndarray
        Input variance (NSPEC, NFIB)
    input_wave : np.ndarray
        Input wavelength axis. Can be 1D (NSPEC,) or 2D (NSPEC, NFIB).
    output_wave : np.ndarray
        Output wavelength axis. Can be 1D (NSPEC_OUT,) or 2D (NSPEC_OUT, NFIB).

    Returns
    -------
    tuple
        (out_spec, out_var) - Rebinned spectra and variance
    """
    nspec, nfib = spectra.shape

    # Check dimensions
    in_wave_2d = input_wave.ndim == 2
    out_wave_2d = output_wave.ndim == 2

    # Determine output shape
    if out_wave_2d:
        nout = output_wave.shape[0]
    else:
        nout = len(output_wave)

    out_spec = np.zeros((nout, nfib), dtype=np.float32)
    out_var = np.zeros((nout, nfib), dtype=np.float32)

    for f in range(nfib):
        # Get x_in for this fiber
        if in_wave_2d:
            x_in = input_wave[:, f]
        else:
            x_in = input_wave

        # Get x_out for this fiber
        if out_wave_2d:
            x_out = output_wave[:, f]
        else:
            x_out = output_wave

        # Interpolate
        # Note: interp1d requires x_in to be strictly increasing?
        # Usually arc solutions are increasing.
        # If not, we might need to sort, but wavelength solutions should be monotonic.
        f_flux = interp1d(x_in, spectra[:, f], kind='linear', bounds_error=False, fill_value=0.0)
        out_spec[:, f] = f_flux(x_out)

        if variance is not None:
             f_var = interp1d(x_in, variance[:, f], kind='linear', bounds_error=False, fill_value=0.0)
             out_var[:, f] = f_var(x_out)

    return out_spec, out_var

def scrunch_from_arc_id(obj_filename, arc_filename, args, reverse=False):
    """
    Scrunches (rebins) the object file using wavelength solution from the arc file.

    If reverse=True, performs the reverse operation:
    Assumes the object file is currently on the linear grid defined by the arc,
    and rebins it back to the original pixel grid of the arc.

    Parameters
    ----------
    obj_filename : str
        Path to the object FITS file to scrunch (modified in place)
    arc_filename : str
        Path to the arc FITS file containing WAVELA extension.
    args : dict
        Arguments dictionary (unused in logic but kept for interface compatibility).
    reverse : bool
        If True, un-scrunch (Linear -> Pixel). Default False (Pixel -> Linear).
    """
    logger.info(f"Scrunching {obj_filename} using {arc_filename} (Reverse={reverse})")

    # Open Object File
    with ImageFile(obj_filename, mode='UPDATE') as obj_file:
        spectra = obj_file.read_image_data().T # (NSPEC, NFIB)
        variance = obj_file.read_variance_data().T
        nx, nf = obj_file.get_size()

        # Open Arc File to get Wavelength Solution
        arc_wave = None
        if arc_filename:
            with ImageFile(arc_filename, mode='READ') as arc_file:
                # Read WAVELA from Arc
                # read_wave_data returns (NFIB, NSPEC) usually based on IO module
                # Let's verify shape by checking.
                # In reduce_arc.py: red_file.write_wave_data(new_wave.T) where new_wave is (NPIX, NFIB).
                # So write_wave_data takes (NFIB, NPIX).
                # read_wave_data returns what write_wave_data wrote.
                # So read_wave_data returns (NFIB, NPIX).
                # We want (NPIX, NFIB) for our logic.
                raw_wave = arc_file.read_wave_data()
                if raw_wave is not None:
                    arc_wave = raw_wave.T
                else:
                    logger.warning(f"No WAVELA found in Arc file {arc_filename}.")

        if arc_wave is None:
            # Unit scaling fallback: 0 to NX-1
            logger.info("Using unit scaling for wavelength.")
            arc_wave = np.zeros((nx, nf))
            for f in range(nf):
                arc_wave[:, f] = np.arange(nx)

        # Determine Global Output Axis (Linear)
        # Defined by min/max of the Arc Solution
        min_wave = np.min(arc_wave)
        max_wave = np.max(arc_wave)

        # Calculate dispersion (approx) from center fiber to determine step size
        center_fib = nf // 2
        center_wave = arc_wave[:, center_fib]
        # Robust dispersion estimate
        disp = (center_wave[-1] - center_wave[0]) / (len(center_wave) - 1)

        # Define linear output axis
        out_axis = np.linspace(min_wave, max_wave, nx)

        # Perform Scrunch
        if not reverse:
            # Forward: Input=Pixel-based Wavelengths (Arc Wave). Output=Linear Axis.
            new_spectra, new_var = rebin_data(spectra, variance, arc_wave, out_axis)

            # Update Object
            obj_file.write_image_data(new_spectra.T)
            obj_file.write_variance_data(new_var.T)
            obj_file.set_header_value("SCRUNCH", True)

        else:
            # Reverse: Input=Linear Axis (out_axis). Output=Pixel-based Wavelengths (Arc Wave).
            # We map FROM the linear grid TO the pixels.
            new_spectra, new_var = rebin_data(spectra, variance, out_axis, arc_wave)

            obj_file.write_image_data(new_spectra.T)
            obj_file.write_variance_data(new_var.T)
            # Typically we don't clear SCRUNCH flag here as this is usually an intermediate step,
            # but if it was the final state, we might. REDUCE_FFLAT handles the final flag.
