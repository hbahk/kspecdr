"""
Multi-Target Tracking (MTT) algorithms.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def multi_target_tracking(
    pk_grid: np.ndarray,
    nsteps: int,
    max_ntraces: int,
    max_displacement: float,
    *,
    min_fraction: float = 0.5,
    gap_limit: Optional[int] = None,
    missing_cost: Optional[float] = None,
    use_float32: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    Link per-step peak detections into traces using a Fortran-like
    Multi-Target Tracking (MTT) approach (PK_GRID2TRACES from 2dfdr) with
    LAP (Hungarian) assignment.

    This is designed to be close in spirit to the Fortran MULTI_TARGET_TRACKING:
      - Unique assignment (1:1) between existing tracks and current-step points.
      - Cost uses Euclidean distance in (x, seq) space:
            cost = sqrt( (dx)^2 + (gap)^2 )
      - Proximity gating:
            abs(dx) <= max_displacement
            gap <= gap_limit  (default: nsteps//4, like Fortran's NSEQ/4)
      - Performance: filters out tracks/points with no possible associations
        before forming the LAP matrix (critical for large fiber counts).

    Parameters
    ----------
    pk_grid : np.ndarray
        Peak grid array (nsteps, >= something). pk_grid[step, j] is peak position
        or 0.0 if absent.
    nsteps : int
        Number of steps (sequences).
    max_ntraces : int
        Maximum number of traces to track/return.
    max_displacement : float
        Maximum allowed |dx| for candidate associations (Fortran MAX_DIST).
    min_fraction : float, optional
        Keep only traces with assigned points > min_fraction * nsteps.
        (Matches PK_GRID2TRACES default behavior when min_fraction=0.5.)
    gap_limit : int, optional
        Maximum allowed gap (current_step - last_step) for candidate association.
        If None, uses nsteps//4 (Fortran uses NSEQ/4).
    missing_cost : float, optional
        Cost for assigning a track to "missing" (dummy) instead of a real point.
        If None, uses slightly above the worst plausible real cost.
    use_float32 : bool, optional
        Use float32 for cost computations to reduce memory and improve speed.

    Returns
    -------
    (ntraces, trace_pts) : Tuple[int, np.ndarray]
        ntraces : int
            Number of significant traces after filtering.
        trace_pts : np.ndarray
            Shape (max_ntraces, nsteps), sorted by median position. Missing = 0.0.
    """

    if pk_grid.shape[0] < nsteps:
        raise ValueError(f"pk_grid has {pk_grid.shape[0]} steps but nsteps={nsteps}")

    if gap_limit is None:
        gap_limit = max(1, nsteps // 4)

    # For missing cost, pick something larger than any valid gated real match.
    # Real matches have |dx|<=max_displacement and gap<=gap_limit, so
    # max real cost <= sqrt(max_displacement^2 + gap_limit^2).
    if missing_cost is None:
        missing_cost = float(np.sqrt(max_displacement**2 + gap_limit**2) * 1.05)

    # Output buffer: we will build tracks here, then filter+sort at the end.
    trace_pts = np.zeros(
        (max_ntraces, nsteps), dtype=np.float32 if use_float32 else np.float64
    )

    # Track state arrays (size max_ntraces; only first ntracks are active)
    last_step = np.full(max_ntraces, -1, dtype=np.int32)
    last_pos = np.zeros(max_ntraces, dtype=trace_pts.dtype)

    # ---- Step 0: Find reliable start sequence ----
    # Strategy: Find step with maximum number of peaks.
    # Break ties by choosing the one closest to the center of the image.
    peak_counts = np.zeros(nsteps, dtype=int)
    for s in range(nsteps):
        peak_counts[s] = np.count_nonzero(pk_grid[s] > 0.0)

    max_count = peak_counts.max()
    if max_count == 0:
        return 0, np.zeros((max_ntraces, nsteps), dtype=float)

    candidates = np.where(peak_counts == max_count)[0]
    center = nsteps / 2.0
    best_idx = np.argmin(np.abs(candidates - center))
    start_seq = candidates[best_idx]

    logger.debug(f"Selected start_seq: {start_seq} with {max_count} peaks")

    # Initialize tracks at start_seq
    peaks = pk_grid[start_seq]
    peaks = peaks[peaks > 0.0]
    peaks = np.sort(peaks.astype(trace_pts.dtype, copy=False))

    n_init = min(peaks.size, max_ntraces)
    trace_pts[:n_init, start_seq] = peaks[:n_init]
    last_step[:n_init] = start_seq
    last_pos[:n_init] = peaks[:n_init]
    ntracks = n_init

    logger.debug(f"Initialized {ntracks} tracks at start_seq")

    # ---- Helper for tracking steps ----
    def process_steps(step_indices):
        nonlocal ntracks

        for s in step_indices:
            peaks = pk_grid[s]
            peaks = peaks[peaks > 0.0]
            if peaks.size == 0:
                continue

            # Sort peaks for deterministic behavior and some cache-friendliness
            peaks = np.sort(peaks.astype(trace_pts.dtype, copy=False))

            # If no existing tracks (possible if max_ntraces==0 or something odd), spawn.
            if ntracks == 0:
                n_add = min(peaks.size, max_ntraces)
                trace_pts[:n_add, s] = peaks[:n_add]
                last_step[:n_add] = s
                last_pos[:n_add] = peaks[:n_add]
                ntracks = n_add
                continue

            # Active track indices
            trk_idx_all = np.arange(ntracks, dtype=np.int32)

            # Compute gap for each track to current sequence
            # Use ABS for bidirectional support
            gaps = np.abs(s - last_step[:ntracks]).astype(np.int32)

            # Candidate tracks must have a valid last_step and satisfy gap_limit
            cand_trk_mask = (last_step[:ntracks] >= 0) & (gaps <= gap_limit)
            if not np.any(cand_trk_mask):
                # No track can associate due to gaps -> all peaks start new tracks (up to capacity)
                n_can_add = max_ntraces - ntracks
                if n_can_add > 0:
                    add = min(peaks.size, n_can_add)
                    trace_pts[ntracks : ntracks + add, s] = peaks[:add]
                    last_step[ntracks : ntracks + add] = s
                    last_pos[ntracks : ntracks + add] = peaks[:add]
                    ntracks += add
                continue

            cand_trk = trk_idx_all[cand_trk_mask]
            cand_last_pos = last_pos[cand_trk]  # shape (m,)
            cand_gaps = gaps[cand_trk]  # shape (m,)

            # ---- Build proximity associations (vectorized) ----
            # dx matrix: shape (m_tracks, n_points)
            dx = peaks[None, :] - cand_last_pos[:, None]
            abs_dx = np.abs(dx)

            # gating by |dx| <= max_displacement
            viable = abs_dx <= max_displacement

            # Count viable associations per track and per point
            t2p_counts = viable.sum(axis=1)
            p2t_counts = viable.sum(axis=0)

            # Tracks/points with at least one viable association are the only ones in LAP
            lap_trk_mask = t2p_counts > 0
            lap_pt_mask = p2t_counts > 0

            # Points with no viable track: they become new tracks immediately
            orphan_peaks = peaks[~lap_pt_mask]

            if orphan_peaks.size:
                n_can_add = max_ntraces - ntracks
                if n_can_add > 0:
                    add = min(orphan_peaks.size, n_can_add)
                    trace_pts[ntracks : ntracks + add, s] = orphan_peaks[:add]
                    last_step[ntracks : ntracks + add] = s
                    last_pos[ntracks : ntracks + add] = orphan_peaks[:add]
                    ntracks += add

            # If no LAP candidates remain, continue to next step
            if not np.any(lap_trk_mask) or not np.any(lap_pt_mask):
                continue

            lap_trk = cand_trk[lap_trk_mask]  # original track indices
            lap_peaks = peaks[lap_pt_mask]  # point positions

            # Reduced dx/viable arrays for LAP
            dx_sub = lap_peaks[None, :] - last_pos[lap_trk][:, None]
            abs_dx_sub = np.abs(dx_sub)

            # Also apply gap gating
            gap_sub = np.abs(s - last_step[lap_trk]).astype(np.int32)
            viable_sub = abs_dx_sub <= max_displacement

            m = lap_trk.size
            n = lap_peaks.size

            # ---- Form LAP cost matrix ----
            dtype_cost = np.float32 if use_float32 else np.float64
            cost = np.full((m, n + m), missing_cost, dtype=dtype_cost)

            if np.any(viable_sub):
                gap_f = gap_sub.astype(dtype_cost, copy=False)[:, None]
                dx_f = dx_sub.astype(dtype_cost, copy=False)
                tmp = np.sqrt(dx_f * dx_f + gap_f * gap_f, dtype=dtype_cost)
                cost[:, :n][viable_sub] = tmp[viable_sub]

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            used_pts = np.zeros(n, dtype=bool)

            # Apply assignments
            for r, c in zip(row_ind, col_ind):
                if r >= m:
                    continue
                if c < n:
                    if cost[r, c] <= missing_cost and abs_dx_sub[r, c] <= max_displacement:
                        trk = lap_trk[r]
                        pos = float(lap_peaks[c])
                        trace_pts[trk, s] = pos
                        last_step[trk] = s
                        last_pos[trk] = pos
                        used_pts[c] = True

            # Any unused LAP peaks become new tracks
            unused_peaks = lap_peaks[~used_pts]
            if unused_peaks.size:
                n_can_add = max_ntraces - ntracks
                if n_can_add > 0:
                    add = min(unused_peaks.size, n_can_add)
                    trace_pts[ntracks : ntracks + add, s] = unused_peaks[:add]
                    last_step[ntracks : ntracks + add] = s
                    last_pos[ntracks : ntracks + add] = unused_peaks[:add]
                    ntracks += add

    # ---- 1. Forward Pass ----
    logger.debug("Starting Forward Pass")
    process_steps(range(start_seq + 1, nsteps))

    # ---- 2. Backward Pass ----
    logger.debug("Starting Backward Pass")

    # Reset state for tracks that existed at start_seq
    # Only these tracks should propagate backwards from start_seq
    # Any tracks spawned during forward pass (index >= n_init) are not valid for backward propagation
    last_step[:n_init] = start_seq
    last_pos[:n_init] = trace_pts[:n_init, start_seq]

    # Disable forward-spawned tracks for the backward pass
    if ntracks > n_init:
        last_step[n_init:ntracks] = -1

    process_steps(range(start_seq - 1, -1, -1))

    # ---- Post-processing (PK_GRID2TRACES-like): filter & sort by median ----
    # Keep traces with enough points
    counts = (trace_pts[:ntracks, :] > 0.0).sum(axis=1)
    keep = counts > (min_fraction * nsteps)

    if not np.any(keep):
        return 0, np.zeros((max_ntraces, nsteps), dtype=float)

    kept_traces = trace_pts[:ntracks, :][keep]

    # Sort by median of nonzero values
    medians = np.empty(kept_traces.shape[0], dtype=np.float64)
    for i in range(kept_traces.shape[0]):
        vals = kept_traces[i][kept_traces[i] > 0.0]
        medians[i] = np.median(vals) if vals.size else 0.0

    order = np.argsort(medians)
    kept_sorted = kept_traces[order]

    nout = min(kept_sorted.shape[0], max_ntraces)
    out = np.zeros((max_ntraces, nsteps), dtype=float)
    out[:nout, :] = kept_sorted[:nout, :].astype(float, copy=False)

    return int(nout), out
