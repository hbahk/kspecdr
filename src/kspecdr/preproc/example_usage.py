"""
Example usage of the MakeIM class for preprocessing raw data to IM format.

This example demonstrates:
1. Basic raw to IM conversion
2. Bias subtraction
3. Dark subtraction  
4. Flat fielding
5. Bad pixel masking
6. Cosmic ray removal
7. Fiber table handling
"""

import logging
from kspecdr.preproc.make_im import MakeIM, make_im

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_conversion():
    """Basic raw to IM conversion."""
    logger.info("=== Basic Raw to IM Conversion ===")
    
    # Create processor
    processor = MakeIM(verbose=True)
    
    # Process raw file to IM
    im_file = processor.process_raw_to_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_im.fits"
    )
    
    logger.info(f"Created IM file: {im_file}")

def example_with_bias_subtraction():
    """Raw to IM conversion with bias subtraction."""
    logger.info("=== Raw to IM with Bias Subtraction ===")
    
    im_file = make_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_bias_im.fits",
        use_bias=True,
        bias_filename="data/bias_taipan.fits"
    )
    
    logger.info(f"Created bias-subtracted IM file: {im_file}")

def example_with_dark_subtraction():
    """Raw to IM conversion with dark subtraction."""
    logger.info("=== Raw to IM with Dark Subtraction ===")
    
    im_file = make_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_dark_im.fits",
        use_dark=True,
        dark_filename="data/dark_taipan.fits"
    )
    
    logger.info(f"Created dark-subtracted IM file: {im_file}")

def example_with_flat_fielding():
    """Raw to IM conversion with flat fielding."""
    logger.info("=== Raw to IM with Flat Fielding ===")
    
    im_file = make_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_flat_im.fits",
        use_flat=True,
        flat_filename="data/flat_taipan.fits"
    )
    
    logger.info(f"Created flat-fielded IM file: {im_file}")

def example_with_bad_pixel_masking():
    """Raw to IM conversion with bad pixel masking."""
    logger.info("=== Raw to IM with Bad Pixel Masking ===")
    
    im_file = make_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_mask_im.fits",
        bad_pixel_mask="data/bad_pixels.fits",
        bad_pixel_mask2="data/bad_pixels2.fits",
        mark_saturated=True
    )
    
    logger.info(f"Created bad-pixel-masked IM file: {im_file}")

def example_with_cosmic_ray_removal():
    """Raw to IM conversion with cosmic ray removal."""
    logger.info("=== Raw to IM with Cosmic Ray Removal ===")
    
    # Using L.A.Cosmic method
    im_file = make_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_cosmic_im.fits",
        cosmic_ray_method="LACOSMIC",
        sigclip=4.5,
        sigfrac=0.3,
        objlim=5.0,
        gain=1.0,
        readnoise=6.5,
        satlevel=65535.0,
        pssl=0.0,
        niter=4,
        sepmed=True,
        cleantype="meanmask",
        fsmode="median",
        psfmodel="gauss",
        psffwhm=2.5,
        psfsize=7,
        psfk=None,
        psfbeta=4.765,
        verbose=False
    )
    
    logger.info(f"Created cosmic-ray-cleaned IM file: {im_file}")

def example_complete_reduction():
    """Complete reduction pipeline."""
    logger.info("=== Complete Reduction Pipeline ===")
    
    im_file = make_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_complete_im.fits",
        use_bias=True,
        bias_filename="data/bias_taipan.fits",
        use_dark=True,
        dark_filename="data/dark_taipan.fits",
        use_flat=True,
        flat_filename="data/flat_taipan.fits",
        bad_pixel_mask="data/bad_pixels.fits",
        mark_saturated=True,
        cosmic_ray_method="LACOSMIC"
    )
    
    logger.info(f"Created complete reduction IM file: {im_file}")

def example_fiber_table_handling():
    """Example showing fiber table handling."""
    logger.info("=== Fiber Table Handling ===")
    
    from kspecdr.io.image import ImageFile
    
    # Process raw file (fiber table will be copied automatically)
    im_file = make_im(
        raw_filename="data/raw_taipan.fits",
        im_filename="data/taipan_fiber_im.fits"
    )
    
    # Check fiber table in output file
    with ImageFile(im_file, mode='READ') as im:
        if im.has_fiber_table():
            fiber_data = im.read_fiber_table()
            table_name = im.get_fiber_table_name()
            logger.info(f"Fiber table '{table_name}' copied with {len(fiber_data)} fibers")
            
            # For TAIPAN, fibers beyond 150 are removed
            if "TAIPAN" in im.get_instrument().upper():
                logger.info("TAIPAN fiber table processed (limited to 150 fibers)")
        else:
            logger.info("No fiber table found in output file")
    
    logger.info(f"Created IM file with fiber table: {im_file}")

if __name__ == "__main__":
    # Run examples
    example_basic_conversion()
    example_with_bias_subtraction()
    example_with_dark_subtraction()
    example_with_flat_fielding()
    example_with_bad_pixel_masking()
    example_with_cosmic_ray_removal()
    example_complete_reduction()
    example_fiber_table_handling()
    
    logger.info("All examples completed!")