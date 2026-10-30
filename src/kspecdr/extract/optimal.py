"""
Optimal (profile-fitting) extraction, port of 2dfdr ``extract_optimal.F95`` (``OPTEX``).

2dfdr ``STANDARD_OPTIMAL_EXTR`` / ``COLUMN_OPTIMAL_EXTR_WTD`` (R. Johnson's CIRPASS method,
"Extracting Closely Packed Spectra") solves, independently for every spectral column, the
weighted linear least-squares problem ``[PHI] eta = d``: the column ``d`` is modelled as the
sum of one fibre profile per fibre (Gaussian of known centre and sigma) times an unknown
amplitude, plus ``NSLPARS`` scattered-light terms (Chebyshev polynomial or B-spline along the
slit). Bad pixels are dropped from the system and the weights are ``1 / VAR``.

This port keeps that model and solves all columns of a chunk at once with the normal
equations (a few tens of parameters per column). Differences from 2dfdr:

- Profiles are **pixel-integrated** Gaussians (exact ``erf`` differences) with the sigma
  measured by :func:`kspecdr.tlm.sigma_profile.sigma_profile` (``SIGMODEL='PIXGAUSS'``);
  2dfdr integrates a point Gaussian with Simpson's rule.
- The variance of an amplitude is the diagonal of ``(PHI^T W PHI)^-1``, which includes the
  covariance with the neighbouring fibres and the scattered-light terms; 2dfdr returns the
  diagonal approximation ``1 / diag(PHI^T W PHI)``.
- The fit uses only the rows between the outer fibres plus a margin, and the scattered-light
  polynomial is normalised over those rows (2dfdr: the full column).
- Optional, off by default: re-weighting with a model-based variance (``niter > 1``, needs
  the gain), rejection of positive outliers (cosmic rays) against that variance plus a
  fractional profile-error term (``clip``), and a piecewise-linear background with knots
  between the fibres (``slmodel='KNOTS'``), the background model of ``sigma_profile``.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.polynomial import chebyshev

from ..tlm.sigma_profile import pixel_gaussian

logger = logging.getLogger(__name__)

SLMODELS = ("POLYN", "KNOTS")
WTSCHEMES = ("STND", "NONE")
MIN_PROFILE_COVERAGE = 0.5  # minimum fraction of a fibre profile on good pixels


def _knot_basis(y: np.ndarray, centres: np.ndarray, margin: float, nk: int) -> np.ndarray:
    """Hat-function basis on knots at the outer margins and between the fibres, (ny, nk)."""
    c = np.sort(centres[np.isfinite(centres)])
    basis = np.zeros((len(y), nk))
    if len(c) == 0:
        return basis
    knots = np.concatenate(([c[0] - margin], 0.5 * (c[1:] + c[:-1]), [c[-1] + margin]))
    eye = np.eye(len(knots))
    for k in range(len(knots)):
        basis[:, k] = np.interp(y, knots, eye[k])
    return basis


def _background_basis(
    y: np.ndarray, centres: np.ndarray, lo: float, hi: float, slmodel: str, nsl: int,
    margin: float,
) -> np.ndarray:
    """Scattered-light basis of one column on rows ``y``, shape (ny, nsl)."""
    if nsl == 0:
        return np.zeros((len(y), 0))
    if slmodel == "KNOTS":
        return _knot_basis(y, centres, margin, nsl)
    t = 2.0 * (y - lo) / max(hi - lo, 1.0) - 1.0
    return chebyshev.chebvander(t, nsl - 1)


def optimal_extract(
    img: np.ndarray,
    var: np.ndarray,
    tlm: np.ndarray,
    sigma: np.ndarray,
    nslpars: int = 1,
    slmodel: str = "POLYN",
    wtscheme: str = "STND",
    gain: Optional[float] = None,
    niter: int = 1,
    clip: float = 0.0,
    prferr: float = 0.02,
    margin: Optional[float] = None,
    chunk: int = 64,
    return_model: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Optimally extract every fibre of an image by fitting all fibre profiles per column.

    Parameters
    ----------
    img, var : np.ndarray
        Image and variance, shape (nspat, nspec), dispersion along x; NaN marks bad pixels.
    tlm : np.ndarray
        Fibre centres (0-based pixels), shape (nfib, nspec). Fibres whose centre is NaN or
        off the detector in a column are not fitted there and return NaN.
    sigma : np.ndarray
        Pixel-integrated Gaussian sigma per fibre and column, shape (nfib, nspec)
        (``SIGMAPRF``, or ``MWIDTH / 2.3548`` everywhere).
    nslpars : int
        Number of scattered-light parameters per column (2dfdr ``OPTEX_NSLPARS``): Chebyshev
        coefficients for ``POLYN`` (0 = none, 1 = constant, 2 = line, ...). Ignored for
        ``KNOTS``, which has one knot per gap plus the two outer margins.
    slmodel : {'POLYN', 'KNOTS'}
        Scattered-light model (2dfdr ``OPTEX_SLMODEL``; ``KNOTS`` is kspecdr-only).
    wtscheme : {'STND', 'NONE'}
        ``STND``: weights 1 / variance; ``NONE``: uniform weights (variances still
        propagated with the input variance).
    gain : float, optional
        Gain (e-/ADU) of the variance model ``var = rn^2 + max(data, 0) / gain``. Needed for
        the model-based variance of iterations after the first.
    niter : int
        Number of fits. The first uses the input variance (2dfdr); later ones replace its
        Poisson part by that of the fitted model and reject outliers when ``clip > 0``.
    clip : float
        Reject pixels whose residual exceeds ``clip`` sigma of the model variance plus
        ``(prferr * profile model)^2``; 0 disables rejection. Only positive outliers are
        rejected (cosmic rays), and only where they are also sharp along the dispersion
        (twice the significance of a neighbouring column), at most one per fibre
        (nearest-centre rows) and column per iteration.
    prferr : float
        Fractional profile-model error used in the rejection threshold, so that the few-%
        mismatch of a Gaussian on bright frames is not taken for cosmic rays.
    margin : float, optional
        Rows fitted beyond the outer fibres, in pixels. Default ``max(pitch / 2, 5 sigma)``.
    chunk : int
        Number of columns solved together.
    return_model : bool
        Also return the model image and the scattered-light image in ``info``.

    Returns
    -------
    flux, fvar : np.ndarray
        Extracted amplitudes (counts per fibre and column) and their variances,
        shape (nfib, nspec); NaN where a fibre could not be solved.
    info : dict
        ``rows`` (first and last fitted row), ``nrej`` (rejected pixels per column),
        ``chi2`` (reduced chi^2 per column against the final variance), ``margin``, and with
        ``return_model`` the ``model`` and ``background`` images (NaN outside the fitted
        rows) and the ``rejected`` pixel mask.
    """
    slmodel = slmodel.upper()
    wtscheme = wtscheme.upper()
    if slmodel not in SLMODELS:
        raise ValueError(f"slmodel must be one of {SLMODELS}, got {slmodel!r}")
    if wtscheme not in WTSCHEMES:
        raise ValueError(f"wtscheme must be one of {WTSCHEMES}, got {wtscheme!r}")
    if niter > 1 and gain is None:
        logger.warning("OPTEX: no gain given, later iterations keep the input variance")
    niter = max(int(niter), 1)

    nspat, nspec = img.shape
    nfib = tlm.shape[0]
    if tlm.shape != (nfib, nspec) or sigma.shape != (nfib, nspec):
        raise ValueError(f"tlm {tlm.shape} and sigma {sigma.shape} must be ({nfib}, {nspec})")

    centres = np.where((tlm > -0.5) & (tlm < nspat - 0.5), tlm, np.nan)
    srt = np.sort(centres, axis=0)
    pitch = np.nanmedian(np.diff(srt, axis=0)) if nfib > 1 else np.nan
    if margin is None:
        sig_med = float(np.nanmedian(sigma))
        margin = max(0.5 * pitch if np.isfinite(pitch) else 0.0, 5.0 * sig_med)
    nsl = nfib + 1 if slmodel == "KNOTS" else max(int(nslpars), 0)

    # One slab of rows covering every column's fit range
    lo_col = np.nanmin(centres, axis=0) - margin
    hi_col = np.nanmax(centres, axis=0) + margin
    y0 = int(max(np.floor(np.nanmin(lo_col)), 0))
    y1 = int(min(np.ceil(np.nanmax(hi_col)), nspat - 1))
    y = np.arange(y0, y1 + 1, dtype=float)

    flux = np.full((nfib, nspec), np.nan)
    fvar = np.full((nfib, nspec), np.nan)
    nrej = np.zeros(nspec, dtype=int)
    chi2 = np.full(nspec, np.nan)
    if return_model:
        model_img = np.full((nspat, nspec), np.nan)
        bg_img = np.full((nspat, nspec), np.nan)
        rej_img = np.zeros((nspat, nspec), dtype=bool)

    # Slab data, one row per column: (nspec, ny)
    D = img[y0 : y1 + 1].T.astype(float)
    V = var[y0 : y1 + 1].T.astype(float)
    inside = (y[None, :] >= lo_col[:, None]) & (y[None, :] <= hi_col[:, None])
    good = np.isfinite(D) & np.isfinite(V) & (V > 0) & inside
    D0 = np.where(good, D, 0.0)
    v_in = np.where(good, V, np.inf)
    w_unif = np.nanmean(np.where(good, 1.0 / v_in, np.nan), axis=1, keepdims=True)
    use = good.copy()
    vfit = v_in
    M = np.zeros_like(D0)  # full model
    Mprof = np.zeros_like(D0)  # fibre profiles only
    if clip > 0:
        # Each row belongs to the fibre with the nearest centre
        cen_all = np.where(np.isfinite(centres), centres, np.inf).T
        owner = np.argmin(np.abs(y[None, :, None] - cen_all[:, None, :]), axis=2)

    chunks = [np.arange(c0, min(c0 + chunk, nspec)) for c0 in range(0, nspec, chunk)]
    chunks = [cols for cols in chunks if good[cols].any()]
    for it in range(niter):
        last = it == niter - 1
        for cols in chunks:
            A, prof, active = _design(y, centres[:, cols].T, sigma[:, cols].T,
                                      lo_col[cols], hi_col[cols], slmodel, nsl, margin)
            w = np.where(use[cols], w_unif[cols] if wtscheme == "NONE" else 1.0 / vfit[cols],
                         0.0)
            eta, cov, ok = _solve(A, w, D0[cols], active)
            M[cols] = np.einsum("cyp,cp->cy", A, eta)
            Mprof[cols] = np.einsum("cyp,cp->cy", A[:, :, :nfib], eta[:, :nfib])
            if not last:
                continue

            u = use[cols]
            if wtscheme == "NONE":
                # Propagate the input variance through the uniform-weight solution
                cov = _sandwich(A, w, u, v_in[cols], cov)
            resid2 = np.where(u, (D0[cols] - M[cols]) ** 2 / np.where(u, vfit[cols], 1.0), 0.0)
            ndof = u.sum(axis=1) - ok.sum(axis=1)
            chi2[cols] = np.where(ndof > 0, resid2.sum(axis=1) / np.maximum(ndof, 1), np.nan)
            nrej[cols] = (good[cols] & ~u).sum(axis=1)

            # Fibres with too little of their profile on good pixels are not trusted
            coverage = np.einsum("cyf,cy->cf", prof, u.astype(float))
            ok_fib = ok[:, :nfib] & (coverage >= MIN_PROFILE_COVERAGE)
            flux[:, cols] = np.where(ok_fib, eta[:, :nfib], np.nan).T
            fv = np.diagonal(cov, axis1=1, axis2=2)[:, :nfib]
            fvar[:, cols] = np.where(ok_fib, fv, np.nan).T

            if return_model:
                fit_rows = inside[cols] & np.any(active, axis=1)[:, None]
                bg = M[cols] - Mprof[cols]
                model_img[y0 : y1 + 1, cols] = np.where(fit_rows, M[cols], np.nan).T
                bg_img[y0 : y1 + 1, cols] = np.where(fit_rows, bg, np.nan).T
                rej_img[y0 : y1 + 1, cols] = (good[cols] & ~u).T
        if last:
            break

        if gain is not None:
            # Replace the Poisson part of the input variance by that of the model
            vfit = v_in + (np.maximum(M, 0.0) - np.maximum(D0, 0.0)) / gain
            vfit = np.where(good, np.maximum(vfit, 1e-6 * np.nanmax(v_in[good])), np.inf)
        if clip > 0:
            use = _reject(D0 - M, vfit, Mprof, good, use, owner, nfib, clip, prferr)

    info = {"rows": (y0, y1), "nrej": nrej, "chi2": chi2, "margin": margin}
    if return_model:
        info["model"] = model_img
        info["background"] = bg_img
        info["rejected"] = rej_img
    nbad = np.sum(~np.isfinite(flux) & np.isfinite(centres))
    logger.info(
        f"OPTEX: {nfib} fibres x {nspec} columns, {nsl} scattered-light terms ({slmodel}), "
        f"rows {y0}-{y1}, {niter} iteration(s), {nrej.sum()} pixels rejected, "
        f"{nbad} unsolved fibre-columns, median chi2 {np.nanmedian(chi2):.2f}"
    )
    return flux, fvar, info


def _design(
    y: np.ndarray, cen: np.ndarray, sig: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    slmodel: str, nsl: int, margin: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Design matrices of a chunk of columns: fibre profiles then scattered light.

    Returns ``A`` (nc, ny, nfib + nsl), the profiles (nc, ny, nfib) and the active-fibre mask
    (nc, nfib).
    """
    nc, nfib = cen.shape
    active = np.isfinite(cen) & np.isfinite(sig) & (sig > 0)
    prof = pixel_gaussian(
        y[None, :, None],
        np.where(active, cen, 0.0)[:, None, :],
        np.where(active, sig, 1.0)[:, None, :],
    )
    prof[~np.broadcast_to(active[:, None, :], prof.shape)] = 0.0
    A = np.zeros((nc, len(y), nfib + nsl))
    A[:, :, :nfib] = prof
    for k in range(nc):
        if active[k].any():
            A[k, :, nfib:] = _background_basis(y, cen[k], lo[k], hi[k], slmodel, nsl, margin)
    return A, prof, active


def _reject(
    resid: np.ndarray, vfit: np.ndarray, mprof: np.ndarray, good: np.ndarray, use: np.ndarray,
    owner: np.ndarray, nfib: int, clip: float, prferr: float,
) -> np.ndarray:
    """
    Update the mask of pixels used in the fit with cosmic-ray rejection, arrays (nspec, ny).

    A new rejection must exceed ``clip`` sigma and twice the significance of the same row in
    the smaller of the two neighbouring columns (cosmic rays are sharp along the dispersion,
    profile mismatch is not); at most one per fibre and column per iteration (Horne 1986),
    as a cosmic ray distorts the whole first fit. Earlier rejections are kept only while they
    still exceed ``clip``.
    """
    z = resid / np.sqrt(vfit + (prferr * mprof) ** 2)
    z = np.where(good, z, -np.inf)
    zpad = np.pad(z, ((1, 1), (0, 0)), constant_values=np.inf)
    zpad[zpad == -np.inf] = np.inf  # a bad neighbour gives no information
    neighbour = np.minimum(zpad[:-2], zpad[2:])
    rejected = good & ~use & (z > clip)
    cand = np.where(good & ~rejected & (z > 2.0 * neighbour), z, -np.inf)
    ncol = z.shape[0]
    for f in range(nfib):
        zf = np.where(owner == f, cand, -np.inf)
        iy = np.argmax(zf, axis=1)
        hit = np.flatnonzero(zf[np.arange(ncol), iy] > clip)
        rejected[hit, iy[hit]] = True
    return good & ~rejected


def _solve(
    A: np.ndarray, w: np.ndarray, d: np.ndarray, active: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched weighted least squares ``min |sqrt(w) (A eta - d)|^2`` per column.

    Parameters without support (inactive fibres, no weighted pixels) are pinned to 0 and
    flagged; returns the solution, its covariance ``(A^T W A)^-1`` and the ok flags.
    """
    nc, _, npar = A.shape
    N = np.einsum("cyi,cy,cyj->cij", A, w, A)
    b = np.einsum("cyi,cy,cy->ci", A, w, d)
    diag = np.diagonal(N, axis1=1, axis2=2)
    scale = np.max(diag, axis=1, keepdims=True)
    ok = diag > 1e-10 * np.where(scale > 0, scale, 1.0)
    ok[:, : active.shape[1]] &= active
    # Pin unsupported parameters: identity row/column, zero right-hand side
    pin = ~ok
    N = np.where(pin[:, :, None] | pin[:, None, :], 0.0, N)
    idx = np.arange(npar)
    N[:, idx, idx] = np.where(pin, 1.0, N[:, idx, idx])
    b = np.where(pin, 0.0, b)
    try:
        cov = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(N)
    eta = np.einsum("cij,cj->ci", cov, b)
    cov = np.where(pin[:, :, None] | pin[:, None, :], 0.0, cov)
    return eta, cov, ok


def _sandwich(
    A: np.ndarray, w: np.ndarray, use: np.ndarray, v: np.ndarray, cov_w: np.ndarray
) -> np.ndarray:
    """Covariance ``C A^T W V W A C`` of a least-squares solution with non-1/V weights."""
    wv = np.where(use, w**2 * np.where(use, v, 0.0), 0.0)
    M = np.einsum("cyi,cy,cyj->cij", A, wv, A)
    return cov_w @ M @ cov_w
