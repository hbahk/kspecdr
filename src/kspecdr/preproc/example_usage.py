"""
Example usage of the make_im module.

This script demonstrates how to use the MakeIM class and make_im function
to process raw astronomical data into IM files.
"""

import logging
from pathlib import Path
from .make_im import MakeIM, make_im

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def example_basic_usage():
    """Example of basic usage with minimal parameters."""
    print("=== Basic Usage Example ===")
    
    # Example raw file path (you would replace this with your actual file)
    raw_filename = "example_raw.fits"
    
    # Basic processing - just create IM file with variance
    if Path(raw_filename).exists():
        im_filename = make_im(
            raw_filename=raw_filename,
            verbose=True
        )
        print(f"Created IM file: {im_filename}")
    else:
        print(f"Raw file {raw_filename} not found - this is just an example")

def example_full_processing():
    """Example of full processing with all options."""
    print("\n=== Full Processing Example ===")
    
    # Example file paths
    raw_filename = "example_raw.fits"
    dark_filename = "master_dark.fits"
    flat_filename = "master_flat.fits"
    bad_pixel_mask = "bad_pixels.fits"
    
    # Check if files exist before processing
    files_to_check = [raw_filename, dark_filename, flat_filename, bad_pixel_mask]
    existing_files = [f for f in files_to_check if Path(f).exists()]
    
    if len(existing_files) > 0:
        print(f"Found existing files: {existing_files}")
        
        # Full processing with all options
        im_filename = make_im(
            raw_filename=raw_filename,
            use_bias=True,
            use_dark=True,
            dark_filename=dark_filename if Path(dark_filename).exists() else None,
            use_flat=True,
            flat_filename=flat_filename if Path(flat_filename).exists() else None,
            bad_pixel_mask=bad_pixel_mask if Path(bad_pixel_mask).exists() else None,
            mark_saturated=True,
            cosmic_ray_method='LACOSMIC',
            verbose=True
        )
        print(f"Created IM file: {im_filename}")
    else:
        print("No example files found - this is just a demonstration")

def example_class_usage():
    """Example using the MakeIM class directly."""
    print("\n=== Class Usage Example ===")
    
    # Create processor instance
    processor = MakeIM(verbose=True)
    
    # Example processing with custom parameters
    raw_filename = "example_raw.fits"
    
    if Path(raw_filename).exists():
        im_filename = processor.process_raw_to_im(
            raw_filename=raw_filename,
            use_bias=True,
            mark_saturated=True,
            cosmic_ray_method='NONE'  # Skip cosmic ray removal for this example
        )
        print(f"Created IM file using class: {im_filename}")
    else:
        print(f"Raw file {raw_filename} not found - this is just an example")

def example_parameter_documentation():
    """Print documentation for the main parameters."""
    print("\n=== Parameter Documentation ===")
    
    doc = """
    make_im() Parameters:
    
    raw_filename (str): Path to the raw input file
    im_filename (str, optional): Path for output IM file (default: derived from raw_filename)
    use_bias (bool): Whether to subtract bias frame (default: False)
    use_dark (bool): Whether to subtract dark frame (default: False)
    dark_filename (str): Path to dark frame file
    use_flat (bool): Whether to divide by long-slit flat field (default: False)
    flat_filename (str): Path to flat field file
    bad_pixel_mask (str): Path to bad pixel mask file
    bad_pixel_mask2 (str): Path to second bad pixel mask file (e.g., cosmic ray mask)
    mark_saturated (bool): Whether to mark saturated pixels as bad (default: True)
    cosmic_ray_method (str): Method for cosmic ray removal:
        - 'NONE': No cosmic ray removal
        - 'LACOSMIC': Use LACosmic algorithm
        - 'BCLEAN': Use BCLEAN algorithm
        - 'PYCOSMIC': Use Python cosmic ray removal
    verbose (bool): Whether to print verbose output (default: True)
    """
    
    print(doc)

if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_full_processing()
    example_class_usage()
    example_parameter_documentation()
    
    print("\n=== Usage Notes ===")
    print("1. The make_im function creates IM files with '_im.fits' suffix")
    print("2. Variance HDU is automatically created and initialized")
    print("3. All processing steps are optional and can be controlled via parameters")
    print("4. The module is based on the Fortran MAKE_IM routine from 2dfdr")
    print("5. Error handling and logging are built into the module") 