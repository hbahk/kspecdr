
import unittest
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from kspecdr.wavecal.landmarks import synchronise_signals

class TestLandmarkConsistency(unittest.TestCase):
    def setUp(self):
        self.npix = 100
        self.nfib = 3
        self.ref_fib = 1
        self.maskv = np.zeros(self.nfib, dtype=bool)
        self.nlm = 3

        # Setup logging
        logging.basicConfig(level=logging.INFO)

    def test_bad_landmarks_rejection(self):
        """Test that bad landmarks (large shift) are rejected and neighbor coeffs used."""
        spectra = np.zeros((self.npix, self.nfib))
        x = np.arange(self.npix)
        # Fiber 1 (Ref): Peak at 50
        spectra[:, 1] = np.exp(-0.5 * (x - 50)**2 / 2**2)
        # Fiber 2: Peak at 50 (Good)
        spectra[:, 2] = np.exp(-0.5 * (x - 50)**2 / 2**2)
        # Fiber 0: Peak at 50 (Physically at 50, but LMR says it's shifted)
        spectra[:, 0] = np.exp(-0.5 * (x - 50)**2 / 2**2)

        lmr = np.zeros((self.nfib, self.nlm))

        # Ref Fiber 1
        lmr[1, :] = [20, 50, 80]
        # Fiber 2
        lmr[2, :] = [20, 50, 80]
        # Fiber 0 (Bad tracking: Shifted +20 pixels)
        lmr[0, :] = [40, 70, 100]

        rebin_spectra = synchronise_signals(
            spectra, self.npix, self.nfib, self.maskv, self.ref_fib, lmr, self.nlm
        )

        # Fiber 0 peak should stay at 50 (using neighbor Identity coeffs)
        # If it used the bad landmarks, it would shift to 30.
        peak_idx = np.argmax(rebin_spectra[:, 0])
        self.assertTrue(abs(peak_idx - 50) < 2, f"Fiber 0 peak at {peak_idx}, expected 50")

    def test_good_landmarks_acceptance(self):
        """Test that good landmarks (small shift) are accepted."""
        spectra = np.zeros((self.npix, self.nfib))
        x = np.arange(self.npix)
        # Fiber 1 (Ref): Peak at 50
        spectra[:, 1] = np.exp(-0.5 * (x - 50)**2 / 2**2)
        # Fiber 0: Peak at 52 (Physically shifted by 2 pixels)
        spectra[:, 0] = np.exp(-0.5 * (x - 52)**2 / 2**2)

        lmr = np.zeros((self.nfib, self.nlm))

        # Ref Fiber 1
        lmr[1, :] = [20, 50, 80]
        # Fiber 0 (Small shift +2 pixels)
        lmr[0, :] = [22, 52, 82]

        rebin_spectra = synchronise_signals(
            spectra, self.npix, self.nfib, self.maskv, self.ref_fib, lmr, self.nlm
        )

        # Fiber 0 peak should be corrected to 50
        peak_idx = np.argmax(rebin_spectra[:, 0])
        self.assertTrue(abs(peak_idx - 50) < 2, f"Fiber 0 peak at {peak_idx}, expected 50")

if __name__ == '__main__':
    unittest.main()
