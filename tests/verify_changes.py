
import numpy as np
from astropy.io import fits
import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))

from kspecdr.io.image import ImageFile
from kspecdr.extract.make_ex import make_ex_from_im

def test_read_wave_data_and_extract():
    print("Setting up test data...")

    # Dimensions
    # Using small dimensions for testing
    # Logic from exploration:
    # ISOPLANE Image: (NSPEC, NSPAT) -> (Rows, Cols) -> (NAXIS2, NAXIS1)
    # TLM: (NFIB, NSPEC) -> (Rows, Cols)? Or matches Image?
    # If we assume TLM matches Image geometry for now, but usually TLM is (NFIB, NSPEC).
    # Let's create a TLM that matches what we expect.

    NSPEC = 100
    NSPAT = 50
    NFIB = 5

    # 1. Create Dummy Image File (ISOPLANE format: Spectral x Spatial)
    # NAXIS2=NSPEC, NAXIS1=NSPAT
    im_data = np.ones((NSPEC, NSPAT), dtype=np.float32)
    im_var = np.ones((NSPEC, NSPAT), dtype=np.float32)

    im_header = fits.Header()
    im_header['INSTRUME'] = 'ISOPLANE'
    im_header['CLASS'] = 'MFFFF'

    hdu_im = fits.PrimaryHDU(data=im_data, header=im_header)
    hdu_var = fits.ImageHDU(data=im_var, name='VARIANCE')

    hdul_im = fits.HDUList([hdu_im, hdu_var])
    im_fname = 'tests/test_im.fits'
    hdul_im.writeto(im_fname, overwrite=True)

    # 2. Create Dummy TLM File
    # TLM Data: (NSPEC, NFIB) usually?
    # But read_image_data reads (nx, ny).
    # In make_ex: nx_tlm, nfib = tlm_file.get_size()
    # If TLM is standard FITS (Spectral on X): NAXIS1=NSPEC, NAXIS2=NFIB.
    # get_size returns (NSPEC, NFIB).
    # read_image_data returns (NFIB, NSPEC).
    # tlm_data.T -> (NSPEC, NFIB).

    # So we create TLM with shape (NFIB, NSPEC)
    tlm_data_arr = np.zeros((NFIB, NSPEC), dtype=np.float32)
    # Set fibers at spatial positions 5, 15, 25, 35, 45
    for i in range(NFIB):
        tlm_data_arr[i, :] = 5.0 + i * 10.0

    tlm_header = fits.Header()
    tlm_header['MWIDTH'] = 2.0

    hdu_tlm = fits.PrimaryHDU(data=tlm_data_arr, header=tlm_header)

    # Add WAVELA extension
    # WAVELA usually matches TLM shape (NFIB, NSPEC)
    wave_arr = np.zeros((NFIB, NSPEC), dtype=np.float32)
    for i in range(NFIB):
        wave_arr[i, :] = np.linspace(4000, 8000, NSPEC)

    hdu_wave = fits.ImageHDU(data=wave_arr, name='WAVELA')

    hdul_tlm = fits.HDUList([hdu_tlm, hdu_wave])
    tlm_fname = 'tests/test_tlm.fits'
    hdul_tlm.writeto(tlm_fname, overwrite=True)

    # 3. Test read_wave_data
    print("Testing read_wave_data...")
    try:
        with ImageFile(tlm_fname, mode='READ') as tlm_file:
            nx, ny = tlm_file.get_size() # Should be (NSPEC, NFIB)
            print(f"TLM Size: {nx}, {ny}")

            # This method doesn't exist yet, expect failure until implemented
            try:
                wave_data = tlm_file.read_wave_data(nx, ny)
                print("read_wave_data success!")
                if wave_data.shape == (ny, nx):
                    print(f"Wave data shape correct: {wave_data.shape}")
                else:
                    print(f"Wave data shape Mismatch: {wave_data.shape} vs {(ny, nx)}")
            except AttributeError:
                print("read_wave_data not implemented yet (Expected)")
            except Exception as e:
                print(f"read_wave_data failed with: {e}")

    except Exception as e:
        print(f"File open failed: {e}")

    # 4. Test make_ex_from_im
    print("Testing make_ex_from_im...")
    ex_fname = 'tests/test_ex.fits'
    args = {
        'IMAGE_FILENAME': im_fname,
        'TLMAP_FILENAME': tlm_fname,
        'EXTRAC_FILENAME': ex_fname,
        'EXTR_OPERATION': 'SUM',
        'SUM_WIDTH': 4.0
    }

    try:
        make_ex_from_im(im_fname, tlm_fname, ex_fname, 'STND', args)
        print("Extraction complete.")

        # Check output for WAVELA
        with fits.open(ex_fname) as hdul_out:
            print("Output HDUs:", [hdu.name for hdu in hdul_out])
            if 'WAVELA' in hdul_out:
                print("WAVELA extension found in output!")
                wdata = hdul_out['WAVELA'].data
                print(f"WAVELA Shape: {wdata.shape}")
                # Expect (NFIB, NSPEC) if not transposed, or transposed?
                # ex_img is written as (NFIB, NSPEC).
                # wave_arr was (NFIB, NSPEC).
                if wdata.shape == (NFIB, NSPEC):
                    print("WAVELA shape correct (NFIB, NSPEC).")
                else:
                    print(f"WAVELA shape {wdata.shape} mismatch.")
            else:
                print("WAVELA extension MISSING in output.")

    except AttributeError:
        print("make_ex_from_im failed likely due to missing read_wave_data (Expected if not implemented)")
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_read_wave_data_and_extract()
