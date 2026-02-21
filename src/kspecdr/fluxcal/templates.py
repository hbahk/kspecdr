"""
BOSZ 2024 stellar template library for kspecdr flux calibration.

Provides:
- :func:`parse_bosz_filename` — extract stellar parameters from filename
- :class:`TemplateLibrary` — indexed access to the downloaded BOSZ subgrid
- :func:`prepare_template` — convolve to instrument resolution and resample
- :func:`resample_spectrum` — general-purpose 1-D resampling

BOSZ file format (``_resam.txt.gz``)
------------------------------------
Two-column ASCII (whitespace-separated, no header), one row per wavelength
point. The shared wavelength grid is in a separate file
(``bosz2024_wave_r10000.txt``).

- Column 0: ``H`` (erg/s/cm²/Å/steradian)
- Column 1: ``C`` (continuum; erg/s/cm²/Å/steradian)

Surface flux: ``F = 4π × H``.  Normalised flux: ``F/C = H/C``.
"""

from __future__ import annotations

import csv
import gzip
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .containers import Spectrum1D, StellarTemplate

logger = logging.getLogger(__name__)

__all__ = [
    "parse_bosz_filename",
    "TemplateLibrary",
    "prepare_template",
    "resample_spectrum",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_LIBRARY_DIR = _REPO_ROOT / "data" / "templates" / "bosz2024"

# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

_BOSZ_RE = re.compile(
    r"bosz2024_"
    r"(?P<atmos>[a-z]{2})_"
    r"t(?P<teff>\d+)_"
    r"g\+(?P<logg>[\d.]+)_"
    r"m(?P<feh>[+-][\d.]+)_"
    r"a(?P<alpha>[+-][\d.]+)_"
    r"c(?P<carbon>[+-][\d.]+)_"
    r"v(?P<vmicro>\d+)_"
    r"(?P<resolution>r\d+)_resam\.txt\.gz$"
)


def parse_bosz_filename(filename: str) -> Optional[Dict]:
    """Parse a BOSZ filename into a parameter dictionary.

    Parameters
    ----------
    filename : str
        Filename (basename only) of a BOSZ ``_resam.txt.gz`` file.

    Returns
    -------
    dict or None
        ``{atmos, teff, logg, feh, alpha_m, carbon_m, vmicro, resolution}``
        with numeric types, or *None* if the filename does not match.
    """
    m = _BOSZ_RE.match(Path(filename).name)
    if m is None:
        return None
    return {
        "atmos":    m.group("atmos"),
        "teff":     int(m.group("teff")),
        "logg":     float(m.group("logg")),
        "feh":      float(m.group("feh")),
        "alpha_m":  float(m.group("alpha")),
        "carbon_m": float(m.group("carbon")),
        "vmicro":   int(m.group("vmicro")),
        "resolution": m.group("resolution"),
    }


# ---------------------------------------------------------------------------
# TemplateLibrary
# ---------------------------------------------------------------------------

class TemplateLibrary:
    """Indexed library of BOSZ 2024 stellar template spectra.

    On first instantiation the library directory is scanned and an index CSV
    is written to ``<library_dir>/index_<resolution>.csv`` for fast subsequent
    starts.

    Parameters
    ----------
    library_dir : str or Path, optional
        Root of the BOSZ subgrid.  Defaults to ``data/templates/bosz2024/``.
    resolution : str
        Resolution subdirectory.  Default ``"r10000"``.
    wave_range : (float, float), optional
        If given, trim the shared wavelength grid (and all loaded spectra)
        to this range in Angstrom, reducing memory.  Default ``(3000, 11000)``.
    """

    def __init__(
        self,
        library_dir: str | Path | None = None,
        resolution: str = "r10000",
        wave_range: Tuple[float, float] = (3000.0, 11000.0),
    ) -> None:
        self.library_dir = Path(library_dir) if library_dir else _DEFAULT_LIBRARY_DIR
        self.resolution = resolution
        self.wave_range = wave_range

        # Shared wavelength grid (trimmed to wave_range)
        wave_file = self.library_dir / f"bosz2024_wave_{resolution}.txt"
        if not wave_file.exists():
            raise FileNotFoundError(
                f"Wavelength grid not found: {wave_file}\n"
                "Run: python -m kspecdr.fluxcal.download_bosz"
            )
        full_wave = np.loadtxt(wave_file)
        if wave_range is not None:
            sel = (full_wave >= wave_range[0]) & (full_wave <= wave_range[1])
            self._wave_slice = sel
        else:
            self._wave_slice = np.ones(len(full_wave), dtype=bool)
        self.wavelength = full_wave[self._wave_slice]

        # Build or load index
        self._index_file = self.library_dir / f"index_{resolution}.csv"
        self._index: List[Dict] = []
        self._build_or_load_index()
        logger.info(
            "TemplateLibrary: %d templates, wave %.0f–%.0f Å (%d pixels)",
            len(self._index),
            self.wavelength[0], self.wavelength[-1], len(self.wavelength),
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _build_or_load_index(self) -> None:
        if self._index_file.exists():
            self._load_index()
            return
        self._scan_and_write_index()

    def _scan_and_write_index(self) -> None:
        spec_dir = self.library_dir / self.resolution
        if not spec_dir.exists():
            raise FileNotFoundError(
                f"Template spectra directory not found: {spec_dir}"
            )
        entries = []
        for gz_path in sorted(spec_dir.rglob("*.txt.gz")):
            params = parse_bosz_filename(gz_path.name)
            if params is None:
                continue
            params["filepath"] = str(gz_path.relative_to(self.library_dir))
            entries.append(params)

        if not entries:
            raise RuntimeError(f"No BOSZ template files found in {spec_dir}")

        # Write CSV index
        fields = [
            "filepath", "atmos", "teff", "logg", "feh",
            "alpha_m", "carbon_m", "vmicro", "resolution",
        ]
        with open(self._index_file, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)

        self._index = entries
        logger.info(
            "Built template index: %d entries → %s",
            len(entries), self._index_file,
        )

    def _load_index(self) -> None:
        entries = []
        with open(self._index_file, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["teff"]     = int(row["teff"])
                row["logg"]     = float(row["logg"])
                row["feh"]      = float(row["feh"])
                row["alpha_m"]  = float(row["alpha_m"])
                row["carbon_m"] = float(row["carbon_m"])
                row["vmicro"]   = int(row["vmicro"])
                entries.append(row)
        self._index = entries

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def grid_params(self) -> np.ndarray:
        """Parameter array, shape ``(N, 4)`` → [Teff, logg, [M/H], [α/M]]."""
        return np.array(
            [[e["teff"], e["logg"], e["feh"], e["alpha_m"]] for e in self._index]
        )

    def query(
        self,
        teff_range: Optional[Tuple[float, float]] = None,
        logg_range: Optional[Tuple[float, float]] = None,
        feh_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict]:
        """Return index entries within the given parameter box.

        Entries are *metadata dicts* (cheap); call :meth:`load_template` to
        read the actual spectral data.
        """
        result = []
        for e in self._index:
            if teff_range and not (teff_range[0] <= e["teff"] <= teff_range[1]):
                continue
            if logg_range and not (logg_range[0] <= e["logg"] <= logg_range[1]):
                continue
            if feh_range and not (feh_range[0] <= e["feh"] <= feh_range[1]):
                continue
            result.append(e)
        return result

    def get_template(
        self,
        teff: float,
        logg: float,
        feh: float,
        alpha_m: Optional[float] = None,
    ) -> StellarTemplate:
        """Nearest-grid-point lookup and load.

        Parameters
        ----------
        teff, logg, feh : float
            Requested stellar parameters.
        alpha_m : float, optional
            If None, defaults to 0.00 (or 0.25 for [M/H] ≤ −0.50).

        Returns
        -------
        StellarTemplate
        """
        if alpha_m is None:
            alpha_m = 0.25 if feh <= -0.50 else 0.00

        target = np.array([teff, logg, feh, alpha_m])
        params = self.grid_params
        # Weighted distance (Teff dominates unless normalised)
        scale = np.array([250.0, 0.5, 0.25, 0.25])  # grid step sizes
        dist = np.sum(((params - target) / scale) ** 2, axis=1)
        best_idx = int(np.argmin(dist))
        return self.load_template(self._index[best_idx])

    def load_template(self, entry: Dict) -> StellarTemplate:
        """Read a template spectrum from disk.

        Parameters
        ----------
        entry : dict
            One element from :attr:`_index` (must have ``"filepath"`` key).

        Returns
        -------
        StellarTemplate
        """
        filepath = self.library_dir / entry["filepath"]
        with gzip.open(filepath, "rt") as fh:
            data = np.loadtxt(fh)

        # Columns: 0 = H (Eddington flux), 1 = C (continuum)
        h_all = data[:, 0]
        c_all = data[:, 1]

        # Trim to wave_range
        h = h_all[self._wave_slice]
        c = c_all[self._wave_slice]

        # Convert Eddington flux to surface flux: F = 4π H
        flux = 4.0 * np.pi * h
        cont = 4.0 * np.pi * c

        return StellarTemplate(
            wavelength=self.wavelength.copy(),
            flux=flux,
            continuum=cont,
            teff=entry["teff"],
            logg=entry["logg"],
            feh=entry["feh"],
            alpha_m=entry["alpha_m"],
            carbon_m=entry["carbon_m"],
            vmicro=float(entry["vmicro"]),
            atmos_model=entry["atmos"],
            source=Path(entry["filepath"]).name,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return (
            f"TemplateLibrary({len(self)} templates, "
            f"wave {self.wavelength[0]:.0f}–{self.wavelength[-1]:.0f} Å)"
        )


# ---------------------------------------------------------------------------
# Template preparation
# ---------------------------------------------------------------------------

def resample_spectrum(
    wave_in: np.ndarray,
    flux_in: np.ndarray,
    wave_out: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Resample a 1-D spectrum onto a new wavelength grid.

    Uses linear interpolation.  Out-of-range pixels are filled with
    *fill_value*.

    Parameters
    ----------
    wave_in : ndarray, shape (M,)
    flux_in : ndarray, shape (M,)
    wave_out : ndarray, shape (N,)
    fill_value : float

    Returns
    -------
    ndarray, shape (N,)
    """
    f = interp1d(
        wave_in, flux_in,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    return f(wave_out)


def prepare_template(
    template: StellarTemplate,
    target_wavelength: np.ndarray,
    instrument_fwhm_angstrom: float,
    fwhm_poly_coeffs: Optional[Sequence[float]] = None,
) -> Spectrum1D:
    """Convolve a template to instrument resolution and resample.

    The BOSZ templates are provided at R = 10,000, which is higher than the
    typical KSPEC instrument resolution (R ~ 3,000–5,000).  This function:

    1. Computes the *additional* broadening needed (quadrature subtraction of
       the BOSZ native width from the requested instrument width).
    2. Applies a Gaussian convolution (constant or wavelength-dependent).
    3. Resamples onto the observed wavelength grid.
    4. Carries the BOSZ continuum through (same operations applied).

    Parameters
    ----------
    template : StellarTemplate
        A template loaded from :class:`TemplateLibrary`.
    target_wavelength : ndarray, shape (N,)
        Observed wavelength grid in Angstrom.
    instrument_fwhm_angstrom : float
        Instrument spectral FWHM in Angstrom (``SPECFWHM`` header keyword).
        Used when *fwhm_poly_coeffs* is None.
    fwhm_poly_coeffs : sequence of float, optional
        Polynomial coefficients giving FWHM(λ) in Angstrom as
        ``FWHM(λ) = c0 + c1*λ + c2*λ² + ...`` (from wavecal ``FWHMPCn``
        header keywords).  If provided, a wavelength-dependent convolution
        is applied.

    Returns
    -------
    Spectrum1D
        Template on the observed grid.  ``variance = 0``, ``mask = True``
        everywhere.  ``meta['continuum']`` contains the resampled continuum.
    """
    tw = template.wavelength
    tf = template.flux.copy()
    tc = template.continuum.copy()

    # Step 1: compute the additional broadening
    # BOSZ at R = 10,000: native FWHM_bosz ≈ λ / R at each pixel
    # We need the quadrature difference
    if fwhm_poly_coeffs is not None:
        tf, tc = _convolve_variable(tw, tf, tc, fwhm_poly_coeffs, r_bosz=10_000)
    else:
        tf, tc = _convolve_constant(tw, tf, tc, instrument_fwhm_angstrom, r_bosz=10_000)

    # Step 2: resample
    flux_out = resample_spectrum(tw, tf, target_wavelength)
    cont_out = resample_spectrum(tw, tc, target_wavelength)

    n = len(target_wavelength)
    return Spectrum1D(
        wavelength=target_wavelength.copy(),
        flux=flux_out,
        variance=np.zeros(n, dtype=float),
        mask=np.ones(n, dtype=bool),
        meta={
            "teff": template.teff,
            "logg": template.logg,
            "feh": template.feh,
            "alpha_m": template.alpha_m,
            "source": template.source,
            "continuum": cont_out,
        },
    )


# ---------------------------------------------------------------------------
# Convolution internals
# ---------------------------------------------------------------------------

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _convolve_constant(
    wave: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    instrument_fwhm: float,
    r_bosz: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply constant-width Gaussian broadening."""
    # Representative wavelength for the middle of the range
    lam_mid = 0.5 * (wave[0] + wave[-1])
    fwhm_bosz = lam_mid / r_bosz
    fwhm_add = np.sqrt(max(instrument_fwhm ** 2 - fwhm_bosz ** 2, 0.0))

    if fwhm_add <= 0:
        return flux, cont

    # Convert to pixel sigma
    dlam = np.median(np.diff(wave))
    sigma_pix = (fwhm_add * _FWHM_TO_SIGMA) / dlam

    return gaussian_filter1d(flux, sigma_pix), gaussian_filter1d(cont, sigma_pix)


def _convolve_variable(
    wave: np.ndarray,
    flux: np.ndarray,
    cont: np.ndarray,
    fwhm_coeffs: Sequence[float],
    r_bosz: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply wavelength-dependent Gaussian broadening.

    For each pixel, the local broadening kernel is computed from the
    polynomial FWHM(λ) minus the native BOSZ resolution in quadrature.
    This is implemented as a loop over wavelength chunks for efficiency.
    """
    fwhm_inst = np.polyval(fwhm_coeffs[::-1], wave)  # polyval wants high→low
    fwhm_bosz = wave / r_bosz
    fwhm_add = np.sqrt(np.maximum(fwhm_inst ** 2 - fwhm_bosz ** 2, 0.0))

    dlam = np.gradient(wave)
    sigma_pix = (fwhm_add * _FWHM_TO_SIGMA) / dlam

    # Variable-sigma convolution: approximate by chunked constant-sigma
    n = len(wave)
    n_chunks = max(1, n // 500)
    flux_out = np.empty_like(flux)
    cont_out = np.empty_like(cont)
    chunk_edges = np.linspace(0, n, n_chunks + 1, dtype=int)

    for i in range(n_chunks):
        s, e = chunk_edges[i], chunk_edges[i + 1]
        # Use wider context to avoid edge artefacts
        ctx_s = max(0, s - 50)
        ctx_e = min(n, e + 50)
        sig = np.median(sigma_pix[s:e])
        if sig > 0.1:
            chunk_f = gaussian_filter1d(flux[ctx_s:ctx_e], sig)
            chunk_c = gaussian_filter1d(cont[ctx_s:ctx_e], sig)
            flux_out[s:e] = chunk_f[s - ctx_s : e - ctx_s]
            cont_out[s:e] = chunk_c[s - ctx_s : e - ctx_s]
        else:
            flux_out[s:e] = flux[s:e]
            cont_out[s:e] = cont[s:e]

    return flux_out, cont_out
