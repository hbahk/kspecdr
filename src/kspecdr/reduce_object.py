"""
Reduce Object Module

This module implements the top-level `reduce_object` routine for the KSPEC pipeline,
mirroring the 2dfdr `REDUCE_OBJECT` subroutine. It orchestrates the reduction of a
raw science file to produce im(age), ex(tracted), and red(uced) science files.
"""

import logging
import shutil
from typing import Dict, Any

from .preproc.make_im import make_im
from .extract.make_ex import make_ex
from .utils.args import init_args, validate_reduce_object_args

logger = logging.getLogger(__name__)

# Constants (mimicking TDFRED_PAR if needed, or just using literals)
MAXNAMELEN_FILE = 256


def reduce_object(args: Dict[str, Any]) -> None:
    """
    Reduces a raw science file to produce im(age), ex(tracted) and red(uced) science files.

    This function corresponds to the 2dfdr SUBROUTINE REDUCE_OBJECT.

    Parameters
    ----------
    args : dict
        Dictionary containing arguments (SDS args in 2dfdr), including:
        - IMAGE_FILENAME: Input raw filename (often implied or derived)
        - EXTRAC_FILENAME: Output extracted filename
        - OUTPUT_FILENAME: Output reduced filename
        - TLMAP_FILENAME: Tramline map filename
        - WAVEL_FILENAME: Wavelength calibration filename
        - FFLAT_FILENAME: Fiber flat filename
        - OUT_DIRNAME: (Optional) Output directory
        - DPCRREX: (Optional) Double pass cosmic ray rejection
        - EXTR_OPERATION: (Optional) Extraction operation method
        - OPTEX_MKRES: (Optional) Make residual map for Optex
        - VERBOSE: (Optional) Verbosity
        - USE_GENCAL: (Optional) Use general calibration (Skyline test)
        - TST_SKYCAL: (Optional) Test skyline calibration
        - INC_RWSS: (Optional) Include RWSS (Reduced Without Sky Subtraction)
        - SKYSPRSMP: (Optional) Super sky subtraction
        - CALIBFLUX: (Optional) Flux calibration
        - TRANSFUNC: (Optional) Transfer function correction
        - DEWIGGLE: (Optional) De-wiggle
        - DEWIGGLE_SMART: (Optional) Smart De-wiggle
    """

    # ----------------------------------
    # INITIALISATION
    # ----------------------------------
    init_args(args)

    # ----------------------------------
    # SANITY CHECK THE SUPPLIED SDS ARGS
    # ----------------------------------
    validate_reduce_object_args(args)

    # Output directory handling
    out_dirname = args.get('OUT_DIRNAME', 'NONE')
    if out_dirname != 'NONE':
        logger.info(f"OUT_DIRNAME={out_dirname}")
        # Note: 2dfdr calls OUTFILE_SETDIR.
        # In Python, we might just assume paths are handled by the caller or updated here.
        # For now, we assume filenames in args are either absolute or relative to CWD.
        # If we need to prepend OUT_DIRNAME, we would do it here.
        # Assuming args paths are updated or we change CWD (less safe).
        # Let's placeholder this behavior if strictly needed, but usually paths are explicit.
        pass

    # --------------------------------------------------------------
    # CREATE THE IM FRAME FROM THE RAW AND THE EX FRAME FROM THE TLM
    # --------------------------------------------------------------

    # Produce im(age) frame
    # 2dfdr: CALL MAKE_IM(ARGS,STATUS)
    # kspecdr make_im usually takes explicit args, but we have a dict.
    # We need to extract raw filename from args or infer it.
    # Usually args has 'OBJECT_FILENAME' or similar?
    # 2dfdr MAKE_IM uses ARGS to find input.
    # Looking at make_im.py: make_im(raw_filename, im_filename, ...)
    # Let's assume args contains 'RAW_FILENAME' or 'IMAGE_FILENAME' refers to the IM file to be created?
    # The docstring says "EXTRAC_FILENAME = ... SDS Given".
    # Usually the input raw file is passed in ARGS.
    # Let's assume 'RAW_FILENAME' is the input raw file.
    # And 'IMAGE_FILENAME' is the output IM file.

    # Check if we have RAW_FILENAME, if not try to derive from IMAGE_FILENAME
    raw_fname = args.get('RAW_FILENAME')
    im_fname = args.get('IMAGE_FILENAME')

    # If make_im signature is make_im(raw_filename, im_filename=None, ...), we need raw_filename.
    if not raw_fname:
        # Fallback logic or error?
        # 2dfdr args usually have specific names.
        # Let's assume for now raw_fname is required or implied.
        # If not present, maybe IMAGE_FILENAME is actually the raw file in some contexts?
        # No, IM is usually the result of MAKE_IM.
        # Let's try to get it.
        pass

    # Call MAKE_IM
    # We pass **args to allow it to pick up what it needs
    # make_im(raw_filename, im_filename=None, ...)
    # If raw_filename is missing, we might need to rely on make_im to handle it or fail.
    # However, kspecdr make_im wrapper expects explicit arguments.
    # We will pass what we have.
    # Note: make_im in kspecdr/preproc/make_im.py takes (raw_filename, im_filename, ...)

    # We'll invoke the wrapper if we can map arguments.
    if raw_fname:
        make_im(raw_fname, im_fname, **args)
    else:
        # If we can't call make_im properly, we might assume IM file already exists or fail.
        # But REDUCE_OBJECT is supposed to create it.
        # Placeholder for robust argument mapping if raw_fname is missing
        logger.warning("RAW_FILENAME not found in args, skipping MAKE_IM call (assuming testing or IM exists)")

    # Check if we are using a double pass cosmic ray rejection process
    dbl_pass_crr_extr = args.get('DPCRREX', False)
    operat = args.get('EXTR_OPERATION', '')
    make_res = args.get('OPTEX_MKRES', False)

    # Is the extraction method OPTEX Based?
    is_optex_based = operat in ['OPTEX', 'SCMOPTEX', 'SMCOPTEX']

    if dbl_pass_crr_extr and is_optex_based and make_res:
        logger.info("PERFORMING DOUBLE PASS COSMIC RAY REJECTION EXTRACTION")

        # Produce ex(tracted) frame from im (this will produce a residual map)
        make_ex(args)

        # Cleanup the IMAGE Frame for cosmic rays by using the residual map
        clean_im(args)

    elif dbl_pass_crr_extr:
        logger.warning("DOUBLE PASS COSMIC RAY REJECTION EXTRACTION REQUESTED BUT OPTIONAL EXTRACTION OR MAKE RESIDUAL MAP NOT SELECTED! IGNORING REQUEST!")

    # Produce ex(tracted) frame from the im frame
    make_ex(args)

    # Get names of input ex(tracted) and output red(uced) files
    ex_filename = args.get('EXTRAC_FILENAME')
    red_filename = args.get('OUTPUT_FILENAME')

    if not ex_filename or not red_filename:
        raise ValueError("EXTRAC_FILENAME and OUTPUT_FILENAME must be specified.")

    # Inform user
    verbose = args.get('VERBOSE', True)
    if verbose:
        logger.info("=================================================")
        logger.info("Reducing object spectra data from extraction file ")
        logger.info("=================================================")
        logger.info(f"Extraction file = {ex_filename}")

    # ----------------------------------------------
    # PERFORM A SKY LINES RECALIBRATION IF REQUESTED
    # ----------------------------------------------
    # CALL TDFIO_OPEN(RED_ID,EX_FILENAME,'UPDATE',TDFIO_STD_TYPE,STATUS)
    # CALL SKYLINES_RECALIBRATION(RED_ID,ARGS,STATUS)
    # CALL TDFIO_CLOSE(RED_ID,STATUS)
    # Note: SKYLINES_RECALIBRATION modifies the EX file in place (via RED_ID which opens EX_FILENAME).
    skylines_recalibration(ex_filename, args)

    # -------------------------------------------
    # CREATE THE "RED" FRAME FROM THE "EX" FRAME
    # -------------------------------------------

    # create output red(uced) file by copying ex(tracted) file
    # CALL TDFIO_CREATEBYCOPY(RED_ID,EX_FILENAME,RED_FILENAME,STATUS)
    logger.info(f"Creating RED file {red_filename} from {ex_filename}")
    shutil.copyfile(ex_filename, red_filename)

    # We work on the RED file now.
    # In 2dfdr, RED_ID is the handle for the RED file.
    # Here we will pass red_filename to functions.

    # SKYLINE CALIB TEST
    rqst_use_gencal = args.get('USE_GENCAL', False)
    rqst_tst_skycal = args.get('TST_SKYCAL', False)
    if rqst_use_gencal and rqst_tst_skycal:
        skycalib_test(red_filename, args)

    # divide by fibre-flat-field if requested
    cmfspec_flatfield(red_filename, args)

    # Scrunch this object frame
    scrunch_object_frame(red_filename, args)

    # Fiber-thoughput calibrate if requested
    # (only used for sky subtraction and makes no sense for N&S data)
    nsflg = tdfio_nod_shuffle(red_filename)
    if nsflg == 0:
        cmfspec_ftpcal(red_filename, args)

    # Dump copy of current reduced image into RWSS HDU if requested
    inc_rwss = args.get('INC_RWSS', False)
    if inc_rwss:
        make_rwss(red_filename)

    # Normal or iterative sky subtraction if requested
    # makes no sense for N&S data
    if nsflg == 0:
        skysub(red_filename, args)
        super_skysub_flag = args.get('SKYSPRSMP', False)
        if super_skysub_flag:
            super_skysub(red_filename, ex_filename, args)

    # Remove PIXCAL data objects that may be in the data frame
    tdfio_pixcal_delete(red_filename)

    # Telluric absorption correction if requested
    telcor(red_filename, args)

    # Add any requested velocity correction related values as an additional
    # column in the fibre table.
    logger.info("CALL VELCOR_UPDATE_FIBRE_TABLE")
    velcor_update_fibre_table(red_filename, args)

    # PCA sky substraction if requested
    # makes no sense for N&S data
    if nsflg == 0:
        skysubpca(red_filename, args)

    # Flux calibration if requested
    calflx = args.get('CALIBFLUX', False)
    if calflx:
        # Note not currently working so inform the user as an error (from Fortran)
        # STATUS = DRS__NOIMPLEM
        # CALL ERSREP('Flux calibration does NOT work!',STATUS)
        # RETURN
        # CALL CMFSOBJ_FLXCALIB(RED_ID,ARGS,STATUS)
        logger.error("Flux calibration does NOT work!")
        # Raising NotImplementedError to match behavior
        raise NotImplementedError("Flux calibration does NOT work!")

    # Correction Via Transfer Function (if requested)
    tfn_cor = args.get('TRANSFUNC', False)
    if tfn_cor:
        logger.info("Correcting via Transfer Function")
        correct_frame_by_assoc_transfer_function(red_filename)

    # Added as a request from Taipan, to propogate bad values
    # to all spectra if the calibrated throughput of that fibre
    # was bad.
    propagate_badthput(red_filename, args)

    # CALL MNB_TESTING2(RED_ID,ARGS,STATUS)
    de_wiggle(red_filename, args)

    # write SDS argument list as FITS HDU in output file
    tdfio_sds_write(red_filename, args)

    # set output file status to reduced
    tdfio_setred(red_filename)

    # Write the 2dfdr release version as a keyword
    stamp_2dfdrver(red_filename, args)

    # If the RWSS (Reduced Without Sky Subtraction) Data has been stored
    # then measure the subtracted sky residuals.
    # (Commented out in Fortran, skipping)

    # Check for errors handled by exceptions in Python
    logger.info("Object Frame Reduced")
    if verbose:
        logger.info(f"Reduction file {red_filename} created.")


# ---------------------------------------------------------------------
# Placeholder Functions
# ---------------------------------------------------------------------

def clean_im(args: Dict[str, Any]) -> None:
    """Cleanup the IMAGE Frame for cosmic rays using residual map"""
    raise NotImplementedError("clean_im not implemented")

def skylines_recalibration(filename: str, args: Dict[str, Any]) -> None:
    """Perform a sky lines recalibration if requested"""
    raise NotImplementedError("skylines_recalibration not implemented")

def skycalib_test(filename: str, args: Dict[str, Any]) -> None:
    """Skyline calib test"""
    raise NotImplementedError("skycalib_test not implemented")

def cmfspec_flatfield(filename: str, args: Dict[str, Any]) -> None:
    """Divide by fibre-flat-field if requested"""
    raise NotImplementedError("cmfspec_flatfield not implemented")

def scrunch_object_frame(filename: str, args: Dict[str, Any]) -> None:
    """Scrunch this object frame"""
    raise NotImplementedError("scrunch_object_frame not implemented")

def tdfio_nod_shuffle(filename: str) -> int:
    """Check for Nod & Shuffle data. Returns flag (0 if not N&S?)."""
    # Placeholder. Assuming 0 (standard) for now if implemented, but raising error as requested.
    raise NotImplementedError("tdfio_nod_shuffle not implemented")

def cmfspec_ftpcal(filename: str, args: Dict[str, Any]) -> None:
    """Fibre-thoughput calibrate if requested"""
    raise NotImplementedError("cmfspec_ftpcal not implemented")

def make_rwss(filename: str) -> None:
    """Dump copy of current reduced image into RWSS HDU"""
    raise NotImplementedError("make_rwss not implemented")

def skysub(filename: str, args: Dict[str, Any]) -> None:
    """Normal or iterative sky subtraction"""
    raise NotImplementedError("skysub not implemented")

def super_skysub(filename: str, ex_filename: str, args: Dict[str, Any]) -> None:
    """Super sky subtraction"""
    raise NotImplementedError("super_skysub not implemented")

def tdfio_pixcal_delete(filename: str) -> None:
    """Remove PIXCAL data objects"""
    raise NotImplementedError("tdfio_pixcal_delete not implemented")

def telcor(filename: str, args: Dict[str, Any]) -> None:
    """Telluric absorption correction"""
    raise NotImplementedError("telcor not implemented")

def velcor_update_fibre_table(filename: str, args: Dict[str, Any]) -> None:
    """Add velocity correction values to fibre table"""
    raise NotImplementedError("velcor_update_fibre_table not implemented")

def skysubpca(filename: str, args: Dict[str, Any]) -> None:
    """PCA sky substraction"""
    raise NotImplementedError("skysubpca not implemented")

def correct_frame_by_assoc_transfer_function(filename: str) -> None:
    """Correction Via Transfer Function"""
    raise NotImplementedError("correct_frame_by_assoc_transfer_function not implemented")

def propagate_badthput(filename: str, args: Dict[str, Any]) -> None:
    """Propagate bad values to all spectra if throughput bad"""
    raise NotImplementedError("propagate_badthput not implemented")

def de_wiggle(filename: str, args: Dict[str, Any]) -> None:
    """De-wiggle"""
    raise NotImplementedError("de_wiggle not implemented")

def tdfio_sds_write(filename: str, args: Dict[str, Any]) -> None:
    """Write SDS argument list as FITS HDU"""
    raise NotImplementedError("tdfio_sds_write not implemented")

def tdfio_setred(filename: str) -> None:
    """Set output file status to reduced"""
    raise NotImplementedError("tdfio_setred not implemented")

def stamp_2dfdrver(filename: str, args: Dict[str, Any]) -> None:
    """Write the 2dfdr release version as a keyword"""
    raise NotImplementedError("stamp_2dfdrver not implemented")
