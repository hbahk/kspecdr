"""
Download BOSZ 2024 stellar template subgrid for kspecdr flux calibration.

Usage
-----
From the repository root::

    python -m kspecdr.fluxcal.download_bosz            # full download
    python -m kspecdr.fluxcal.download_bosz --dry-run  # preview file list only
    python -m kspecdr.fluxcal.download_bosz --force    # re-download existing files

Subgrid definition (F-type spectrophotometric standards)
---------------------------------------------------------
- Resolution: R = 10,000
- Teff: 5000-8000 K, step 250 K (13 values)
- log(g): 3.5, 4.0, 4.5, 5.0
- [M/H]: -1.00 to +0.50, step 0.25 (7 values)
- [alpha/M]: +0.00 (all [M/H]); additionally +0.25 for [M/H] <= -0.50
- [C/M]: +0.00
- vmicro: 1 km/s
- atmos: mp (MARCS plane-parallel) for Teff 5000-7250 K,
  ap (ATLAS9 plane-parallel) for Teff 7500-8000 K

Output
------
Files are saved to::

    <repo>/data/templates/bosz2024/r10000/<metallicity>/<filename>.txt.gz

The shared wavelength grid is saved to::

    <repo>/data/templates/bosz2024/bosz2024_wave_r10000.txt

References
----------
- Meszaros et al. 2024: The updated BOSZ synthetic stellar spectral library
- MAST HLSP BOSZ: https://archive.stsci.edu/hlsp/bosz
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file: src/kspecdr/fluxcal/download_bosz.py
# Repo root:  ../../../../ → four .parent calls
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
TEMPLATE_ROOT = _REPO_ROOT / "data" / "templates" / "bosz2024"

# ---------------------------------------------------------------------------
# Subgrid definition
# ---------------------------------------------------------------------------

BASE_URL   = "https://archive.stsci.edu/hlsps/bosz/bosz2024"
RESOLUTION = "r10000"

TEFF_LIST  = list(range(5000, 8001, 250))   # 5000–8000 K, step 250
LOGG_LIST  = [3.5, 4.0, 4.5, 5.0]
FEH_LIST   = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50]
VMICRO     = 1
CARBON     = 0.00

MAX_WORKERS = 8
RETRY_MAX   = 3
RETRY_DELAY = 2.0   # seconds


def alpha_values_for(feh: float) -> list[float]:
    """Return [α/M] values to include for a given [M/H]."""
    return [0.00, 0.25] if feh <= -0.50 else [0.00]


def atmos_for(teff: int) -> str:
    """ATLAS9 plane-parallel for Teff ≥ 7500 K; MARCS plane-parallel otherwise."""
    return "ap" if teff >= 7500 else "mp"


# ---------------------------------------------------------------------------
# Filename / URL helpers
# ---------------------------------------------------------------------------

def _fmt_logg(v: float) -> str:
    return f"g+{v:.1f}"

def _fmt_feh(v: float) -> str:
    return f"m{v:+.2f}"

def _fmt_alpha(v: float) -> str:
    return f"a{v:+.2f}"

def _fmt_carbon(v: float) -> str:
    return f"c{v:+.2f}"

def _fmt_teff(v: int) -> str:
    return f"t{v:d}"


def build_filename(atmos, teff, logg, feh, alpha, carbon, vmicro, resolution):
    return (
        f"bosz2024_{atmos}_{_fmt_teff(teff)}_{_fmt_logg(logg)}"
        f"_{_fmt_feh(feh)}_{_fmt_alpha(alpha)}_{_fmt_carbon(carbon)}"
        f"_v{vmicro}_{resolution}_resam.txt.gz"
    )


def build_url(feh: float, filename: str) -> str:
    return f"{BASE_URL}/{RESOLUTION}/{_fmt_feh(feh)}/{filename}"


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

def _download_one(
    url: str,
    dest: Path,
    skip_existing: bool = True,
) -> tuple[str, Path, str]:
    """
    Download *url* to *dest*.

    Returns
    -------
    (url, dest, status)  where status ∈ {'ok', 'skipped', 'not_found', 'error:<msg>'}
    """
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return url, dest, "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, RETRY_MAX + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return url, dest, "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return url, dest, "not_found"
            if attempt < RETRY_MAX:
                time.sleep(RETRY_DELAY)
            else:
                return url, dest, f"error:HTTP{exc.code}"
        except Exception as exc:           # noqa: BLE001
            if attempt < RETRY_MAX:
                time.sleep(RETRY_DELAY)
            else:
                return url, dest, f"error:{exc}"

    return url, dest, "error:exhausted"   # unreachable, satisfies type checker


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------

def build_task_list() -> list[tuple[str, Path]]:
    tasks = []
    for teff in TEFF_LIST:
        atmos = atmos_for(teff)
        for logg in LOGG_LIST:
            for feh in FEH_LIST:
                for alpha in alpha_values_for(feh):
                    fname = build_filename(
                        atmos, teff, logg, feh, alpha, CARBON, VMICRO, RESOLUTION
                    )
                    url  = build_url(feh, fname)
                    dest = TEMPLATE_ROOT / RESOLUTION / _fmt_feh(feh) / fname
                    tasks.append((url, dest))
    return tasks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, skip_existing: bool = True) -> None:
    TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)

    # --- wavelength grid (shared across all models at this resolution) ---
    wave_url  = f"{BASE_URL}/wavelength_grids/bosz2024_wave_{RESOLUTION}.txt"
    wave_dest = TEMPLATE_ROOT / f"bosz2024_wave_{RESOLUTION}.txt"

    if wave_dest.exists() and wave_dest.stat().st_size > 0:
        print(f"  wavelength grid : already exists — {wave_dest.name}")
    else:
        print(f"  wavelength grid : {wave_dest.relative_to(_REPO_ROOT)}")
        if not dry_run:
            _, _, status = _download_one(wave_url, wave_dest, skip_existing=False)
            print(f"    → {status}")

    # --- model spectra ---
    tasks = build_task_list()

    feh_alpha_pairs = set(
        (_fmt_feh(feh), _fmt_alpha(a))
        for feh in FEH_LIST for a in alpha_values_for(feh)
    )

    print("\nSubgrid summary")
    print(f"  Teff    : {TEFF_LIST[0]}–{TEFF_LIST[-1]} K, step 250 ({len(TEFF_LIST)} values)")
    print(f"  logg    : {LOGG_LIST}")
    print(f"  [M/H]   : {[f'{f:+.2f}' for f in FEH_LIST]}")
    print("  [α/M]   : 0.00 for all; +0.25 additionally for [M/H] ≤ −0.50")
    print("  atmos   : mp (Teff 5000–7250 K) / ap (Teff 7500–8000 K)")
    print(f"  Total   : {len(tasks)} model files")
    print(f"  Dest    : {TEMPLATE_ROOT.relative_to(_REPO_ROOT)}/")

    if dry_run:
        print("\nDry run — first 8 URLs:")
        for url, _ in tasks[:8]:
            print(f"  {url}")
        print(f"  ... ({len(tasks) - 8} more)")
        return

    print()
    counts: dict[str, int] = {"ok": 0, "skipped": 0, "not_found": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_download_one, url, dest, skip_existing): (url, dest)
            for url, dest in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            url, dest, status = future.result()
            key = status if status in counts else "error"
            counts[key] += 1
            print(f"  [{i:>4}/{len(tasks)}] {status:<16}  {dest.name}", flush=True)

    print("\n--- Summary ---")
    for k, v in counts.items():
        print(f"  {k:>10}: {v}")
    if counts["not_found"]:
        print(
            "\n  Note: 'not_found' entries are expected for (Teff, logg) combinations\n"
            "  that lie outside the BOSZ grid boundaries."
        )


if __name__ == "__main__":
    _dry  = "--dry-run" in sys.argv
    _force = "--force" in sys.argv
    main(dry_run=_dry, skip_existing=not _force)
