"""
Fibre profile widths from a fibre flat (port of 2dfdr ``sigma_profile.F95``).

2dfdr ``MAKE_TLM`` with ``SIGPROF=TRUE`` measures a Gaussian sigma for every fibre and column
of an MFFFF frame and stores it in the ``SIGMAPRF`` HDU of the tramline map, which optimal
extraction (``OPTEX``) reads instead of the constant ``MWIDTH``. The 2dfdr routine fits
column-median slices with sampled Gaussians, a scaled block-boundary background and
leave-one-out FWHM/area corrections, then a robust quadratic per fibre along the dispersion.

This port keeps that structure (median slices every ``col_step`` columns, per-fibre
polynomial along the dispersion) but replaces the heuristic slice fit by a joint
least-squares fit of **pixel-integrated** Gaussians (amplitude, centre offset and sigma per
fibre) on a piecewise-linear background with knots between the fibres. The sigma is
therefore that of the profile before pixel sampling, the quantity a pixel-integrated
extraction model needs; a sampled-Gaussian sigma is larger by about ``1/(24 sigma)``.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

logger = logging.getLogger(__name__)

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
SIGMA_MIN = 0.3  # px, lower bound of the slice fit


def pixel_gaussian(y: np.ndarray, centre: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Fraction of a unit-area Gaussian falling in unit pixels centred on ``y``.

    Parameters
    ----------
    y : np.ndarray
        Pixel centres (0-based index coordinates).
    centre, sigma : np.ndarray
        Gaussian centre and sigma in pixels, broadcastable against ``y``.

    Returns
    -------
    np.ndarray
        ``Phi((y + 0.5 - centre) / sigma) - Phi((y - 0.5 - centre) / sigma)``.
    """
    s = np.sqrt(2.0) * sigma
    return 0.5 * (erf((y + 0.5 - centre) / s) - erf((y - 0.5 - centre) / s))


def _median_slice(
    img: np.ndarray, var: Optional[np.ndarray], lo: int, hi: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Median over columns ``lo:hi`` and its 1-sigma error (NaN where no good pixel)."""
    block = img[:, lo:hi]
    ngood = np.sum(np.isfinite(block), axis=1)
    with np.errstate(all="ignore"):
        data = np.nanmedian(block, axis=1) if np.any(ngood) else np.full(len(block), np.nan)
        if var is not None:
            # Variance of a median of n Gaussian samples: pi/2 * var / n
            err = np.sqrt(0.5 * np.pi * np.nanmean(var[:, lo:hi], axis=1) / ngood)
        else:
            err = np.ones(len(block))
    bad = (ngood == 0) | ~np.isfinite(err) | (err <= 0)
    data[bad] = np.nan
    err[bad] = np.nan
    return data, err


def _fit_slice(
    data: np.ndarray, err: np.ndarray, centres: np.ndarray, sigma0: float, sigma_max: float
) -> Dict[str, np.ndarray]:
    """
    Joint fit of pixel-integrated Gaussians and a piecewise-linear background to one slice.

    Parameters
    ----------
    data, err : np.ndarray
        Slice values and errors along the slit, shape (nspat,); NaN marks bad pixels.
    centres : np.ndarray
        Tramline centres of the fibres to fit (0-based pixels), any order.
    sigma0 : float
        Starting sigma.
    sigma_max : float
        Upper bound of sigma.

    Returns
    -------
    dict
        ``flux``, ``offset``, ``sigma`` and their errors ``eflux``, ``esigma`` per fibre (in
        the order of ``centres``); ``y``, ``model`` and ``background`` on the fitted pixels;
        and ``ok`` (bool) for the whole fit.
    """
    nf = len(centres)
    nspat = len(data)
    order = np.argsort(centres)
    c = centres[order]
    gaps = np.diff(c)
    pitch = np.median(gaps) if nf > 1 else 4.0 * sigma0
    halfwin = max(0.5 * pitch, 5.0 * sigma0)
    knots = np.concatenate(([c[0] - halfwin], 0.5 * (c[1:] + c[:-1]), [c[-1] + halfwin]))

    y0 = max(int(np.floor(knots[0])), 0)
    y1 = min(int(np.ceil(knots[-1])), nspat - 1)
    y = np.arange(y0, y1 + 1, dtype=float)
    good = np.isfinite(data[y0 : y1 + 1]) & np.isfinite(err[y0 : y1 + 1])
    y, d, e = y[good], data[y0 : y1 + 1][good], err[y0 : y1 + 1][good]
    nk = len(knots)

    # Piecewise-linear background basis (hat functions on the knots)
    basis = np.zeros((len(y), nk))
    for k in range(nk):
        hat = np.zeros(nk)
        hat[k] = 1.0
        basis[:, k] = np.interp(y, knots, hat, left=hat[0], right=hat[-1])

    # Starting values: background from local minima, flux from the peak above it
    b0 = np.array([np.nanmin(data[max(int(kn) - 2, 0) : int(kn) + 3]) if
                   0 <= int(kn) < nspat else np.nan for kn in knots])
    b0 = np.where(np.isfinite(b0), b0, np.nanmedian(d) if len(d) else 0.0)
    bg_at = np.interp(c, knots, b0)
    peak = np.array([data[int(round(ci))] if 0 <= int(round(ci)) < nspat else np.nan for ci in c])
    f0 = np.nan_to_num((peak - bg_at) * np.sqrt(2.0 * np.pi) * sigma0, nan=0.0)

    p0 = np.concatenate((f0, np.zeros(nf), np.full(nf, sigma0), b0))
    lo = np.concatenate((np.full(nf, -np.inf), np.full(nf, -1.0), np.full(nf, SIGMA_MIN),
                         np.full(nk, -np.inf)))
    hi = np.concatenate((np.full(nf, np.inf), np.full(nf, 1.0), np.full(nf, sigma_max),
                         np.full(nk, np.inf)))
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    def residual(p):
        flux, off, sig, bg = p[:nf], p[nf : 2 * nf], p[2 * nf : 3 * nf], p[3 * nf :]
        prof = pixel_gaussian(y[:, None], (c + off)[None, :], sig[None, :])
        return (prof @ flux + basis @ bg - d) / e

    def jacobian(p):
        flux, off, sig = p[:nf], p[nf : 2 * nf], p[2 * nf : 3 * nf]
        up = (y[:, None] + 0.5 - (c + off)[None, :]) / sig[None, :]
        um = (y[:, None] - 0.5 - (c + off)[None, :]) / sig[None, :]
        php = np.exp(-0.5 * up**2) / np.sqrt(2.0 * np.pi)
        phm = np.exp(-0.5 * um**2) / np.sqrt(2.0 * np.pi)
        prof = pixel_gaussian(y[:, None], (c + off)[None, :], sig[None, :])
        d_off = -(php - phm) / sig[None, :] * flux[None, :]
        d_sig = -(php * up - phm * um) / sig[None, :] * flux[None, :]
        return np.hstack((prof, d_off, d_sig, basis)) / e[:, None]

    out = {key: np.full(nf, np.nan) for key in ("flux", "offset", "sigma", "eflux", "esigma")}
    out["ok"] = False
    out["y"] = y
    if len(y) <= 3 * nf + nk:
        return out
    try:
        res = least_squares(residual, p0, jac=jacobian, bounds=(lo, hi), x_scale="jac",
                            method="trf", max_nfev=100)
    except (ValueError, np.linalg.LinAlgError):
        return out
    if not res.success:
        return out

    jac = res.jac
    chi2 = np.sum(res.fun**2) / max(len(y) - len(res.x), 1)
    cov = np.linalg.pinv(jac.T @ jac) * max(chi2, 1.0)
    perr = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    inv = np.empty(nf, dtype=int)
    inv[order] = np.arange(nf)
    out["flux"] = res.x[:nf][inv]
    out["offset"] = res.x[nf : 2 * nf][inv]
    out["sigma"] = res.x[2 * nf : 3 * nf][inv]
    out["eflux"] = perr[:nf][inv]
    out["esigma"] = perr[2 * nf : 3 * nf][inv]
    out["background"] = basis @ res.x[3 * nf :]
    out["model"] = res.fun * e + d
    out["ok"] = True
    return out


def _robust_polyfit(
    x: np.ndarray, z: np.ndarray, w: np.ndarray, order: int, clip: float, niter: int = 5
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Weighted polynomial fit with iterative sigma clipping; returns (coeffs, kept mask)."""
    keep = np.ones(len(x), dtype=bool)
    coef = None
    for _ in range(niter):
        if keep.sum() < order + 2:
            return None, keep
        coef = np.polyfit(x[keep], z[keep], order, w=w[keep])
        r = z - np.polyval(coef, x)
        scale = 1.4826 * np.median(np.abs(r[keep] * w[keep]))
        if not np.isfinite(scale) or scale == 0:
            break
        new = np.abs(r * w) < clip * scale
        if np.array_equal(new, keep):
            break
        keep = new
    return coef, keep


def sigma_profile(
    img: np.ndarray,
    tlm: np.ndarray,
    var: Optional[np.ndarray] = None,
    measured: Optional[np.ndarray] = None,
    col_width: int = 11,
    col_step: int = 20,
    min_snr: float = 10.0,
    max_sigerr: float = 0.05,
    order: int = 3,
    clip: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Gaussian sigma of every fibre profile at every column of a fibre flat.

    Port of 2dfdr ``SIGMA_PROFILE`` (``sigma_profile.F95``); see the module docstring for
    the differences.

    Parameters
    ----------
    img : np.ndarray
        Flat image, shape (nspat, nspec), dispersion along x; NaN marks bad pixels.
    tlm : np.ndarray
        Tramline map, shape (nfib, nspec), 0-based pixel centres.
    var : np.ndarray, optional
        Variance of ``img``; without it the slices are fitted unweighted.
    measured : np.ndarray, optional
        Bool (nfib,): fibres whose trace was measured. The others (interpolated tramlines,
        dead fibres) are fitted along with their neighbours but get the slit-median sigma.
    col_width : int, optional
        Columns combined (median) into each slice (2dfdr: 5).
    col_step : int, optional
        Column spacing of the slices (2dfdr: 200).
    min_snr : float, optional
        Smallest fitted flux / error of a slice measurement that is used; slices where no
        fibre can reach it are not fitted.
    max_sigerr : float, optional
        Largest relative sigma error of a slice measurement that is used.
    order : int, optional
        Polynomial order of sigma along the dispersion per fibre (2dfdr: 2; a quadratic
        misses the upturn at the red end of the Isoplane frames).
    clip : float, optional
        Rejection threshold (robust sigma) of the polynomial fit.

    Returns
    -------
    sigma : np.ndarray
        Shape (nfib, nspec). Each fibre's polynomial is held constant beyond its first and
        last accepted slice. Fibres without enough accepted slices take the median over the
        fitted fibres column by column.
    info : dict
        Slice measurements: ``cols`` (nslice,), ``sigma``, ``esigma``, ``offset``, ``snr``
        and ``used`` (each (nfib, nslice)); ``fitted`` (nfib,) bool; ``mwidth``, the median
        FWHM in pixels of the used measurements.

    Raises
    ------
    ValueError
        If no slice measurement passes the cuts.
    """
    nspat, nspec = img.shape
    nfib = tlm.shape[0]
    if measured is None:
        measured = np.ones(nfib, dtype=bool)
    half = col_width // 2
    cols = np.arange(max(col_step // 2, half), nspec - half, col_step)
    nsl = len(cols)

    shape = (nfib, nsl)
    sig, esig, off, snr = (np.full(shape, np.nan) for _ in range(4))
    sigma0 = 1.5
    for i, col in enumerate(cols):
        centres = tlm[:, col]
        inside = np.isfinite(centres) & (centres > 1) & (centres < nspat - 2)
        if inside.sum() == 0:
            continue
        pitch = np.median(np.diff(np.sort(centres[inside]))) if inside.sum() > 1 else 10.0
        data, err = _median_slice(img, var, col - half, col + half + 1)
        # Skip slices where no fibre can reach min_snr (peak S/N x ~sqrt(4 sqrt(pi) sigma)
        # is roughly the flux S/N of a Gaussian)
        rc = np.clip(np.round(centres[inside]).astype(int), 0, nspat - 1)
        with np.errstate(invalid="ignore"):
            peak_snr = (data[rc] - np.nanpercentile(data, 5)) / err[rc]
        if not np.nanmax(peak_snr, initial=-np.inf) * np.sqrt(4.0 * np.sqrt(np.pi) * sigma0) > 0.5 * min_snr:
            continue
        fit = _fit_slice(data, err, centres[inside], sigma0, max(0.45 * pitch, 2.0 * SIGMA_MIN))
        if not fit["ok"]:
            continue
        sig[inside, i] = fit["sigma"]
        esig[inside, i] = fit["esigma"]
        off[inside, i] = fit["offset"]
        with np.errstate(divide="ignore", invalid="ignore"):
            snr[inside, i] = fit["flux"] / fit["eflux"]
        ok_now = inside.copy()
        ok_now[inside] = fit["flux"] / np.maximum(fit["eflux"], 1e-30) > min_snr
        if np.any(ok_now & measured):
            sigma0 = float(np.nanmedian(sig[ok_now & measured, i]))

    with np.errstate(invalid="ignore"):
        used = (
            measured[:, None]
            & np.isfinite(sig)
            & np.isfinite(esig)
            & (esig > 0)
            & (snr > min_snr)
            & (esig < max_sigerr * sig)
            & (np.abs(off) < 0.5)
            & (sig > SIGMA_MIN * 1.01)
        )
    if not np.any(used):
        raise ValueError("No fibre profile measurement passed the cuts")

    xnorm = (np.arange(nspec) - 0.5 * (nspec - 1)) / (0.5 * (nspec - 1))
    xs = xnorm[cols]
    sigma = np.full((nfib, nspec), np.nan)
    fitted = np.zeros(nfib, dtype=bool)
    for f in range(nfib):
        u = used[f]
        if u.sum() < order + 2:
            continue
        coef, keep = _robust_polyfit(xs[u], sig[f, u], 1.0 / esig[f, u], order, clip)
        if coef is None:
            continue
        kept = np.flatnonzero(u)[keep]
        used[f, np.flatnonzero(u)[~keep]] = False
        xlo, xhi = xs[kept.min()], xs[kept.max()]
        sigma[f] = np.polyval(coef, np.clip(xnorm, xlo, xhi))
        fitted[f] = True

    if np.any(fitted):
        fill = np.median(sigma[fitted], axis=0)
    else:
        fill = np.full(nspec, np.nanmedian(sig[used]))
    sigma[~fitted] = fill

    mwidth = float(FWHM_PER_SIGMA * np.median(sig[used]))
    logger.info(
        "Profile sigma: %d slices, %d/%d fibres fitted, median %.3f px (FWHM %.2f px), "
        "range %.3f-%.3f px",
        nsl, int(fitted.sum()), nfib, mwidth / FWHM_PER_SIGMA, mwidth,
        float(np.min(sigma)), float(np.max(sigma)),
    )
    info = {
        "cols": cols, "sigma": sig, "esigma": esig, "offset": off, "snr": snr,
        "used": used, "fitted": fitted, "mwidth": mwidth,
    }
    return sigma, info
