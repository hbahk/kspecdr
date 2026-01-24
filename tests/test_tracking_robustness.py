
import numpy as np
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath('src'))

from kspecdr.tracking import multi_target_tracking

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mtt_initialization():
    print("Testing MTT initialization logic...")

    nsteps = 50
    max_ntraces = 5
    max_displacement = 4.0

    # Create pk_grid
    # pk_grid shape: (nsteps, some_large_buffer) usually, but logic handles it.
    # The input to multi_target_tracking is (nsteps, max_detected_peaks_per_step) technically,
    # or just needs to be large enough to hold peaks.
    # In tracking.py: peaks = pk_grid[s]; peaks = peaks[peaks > 0.0]
    # So we can make it (nsteps, 20).
    pk_grid = np.zeros((nsteps, 20), dtype=float)

    # --- Scenario ---
    # Steps 0-9: "Weak/Noisy" region.
    # 2 "noise" peaks at positions that don't match the real traces.
    # If the algorithm starts here, it will initialize these as tracks.
    for s in range(0, 10):
        pk_grid[s, 0] = 500.0  # Noise far away
        pk_grid[s, 1] = 600.0  # Noise far away

    # Steps 10-49: "Strong" region.
    # 5 real traces at 10, 20, 30, 40, 50.
    real_positions = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    for s in range(10, 50):
        # Add some jitter to make it realistic tracking
        jitter = np.sin(s/10.0)
        current_pos = real_positions + jitter
        pk_grid[s, :5] = current_pos

    print("Running multi_target_tracking...")
    ntraces, trace_pts = multi_target_tracking(
        pk_grid,
        nsteps,
        max_ntraces,
        max_displacement,
        min_fraction=0.1 # Allow shorter traces for this test if needed
    )

    print(f"Found {ntraces} traces.")

    # Analyze results
    # We expect to find 5 traces corresponding to the real positions (approx 10, 20, 30, 40, 50).
    # If we found the noise traces (500, 600), that's bad (or at least not what we want if they blocked the real ones).
    # In this specific setup:
    # - Current logic: Starts at Step 0. Initializes 2 tracks (500, 600).
    #   At Step 10, sees 5 peaks (10..50). They don't match 500/600.
    #   Spawns 3 NEW tracks (filling max_ntraces=5).
    #   So we get 2 noise tracks + 3 real tracks. 2 real tracks are lost.
    # - New logic: Should start at Step 10 (5 peaks > 2 peaks).
    #   Initializes 5 tracks (10..50).
    #   Tracks forward (10->49). All good.
    #   Tracks backward (10->0). Will find nothing matching in 0-9 (noise is at 500).
    #   Result: 5 real tracks found.

    # Check the median positions of the found traces
    found_medians = []
    for i in range(ntraces):
        # Calculate median of valid points
        valid = trace_pts[i, :] > 0
        if np.any(valid):
            med = np.median(trace_pts[i, valid])
            found_medians.append(med)

    found_medians = np.sort(found_medians)
    print(f"Found trace medians: {found_medians}")

    # Check if we found the 5 real traces
    # Real medians should be close to 10, 20, 30, 40, 50
    expected_medians = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # We'll count how many real traces we recovered (within tolerance)
    recovered_count = 0
    for exp in expected_medians:
        # Check if any found median is close
        if np.any(np.abs(found_medians - exp) < 5.0):
            recovered_count += 1

    print(f"Recovered {recovered_count}/5 real traces.")

    if recovered_count < 5:
        print("FAIL: Did not recover all real traces.")
        # Identify if noise tracks were kept
        noise_count = 0
        for med in found_medians:
            if med > 400:
                noise_count += 1
        print(f"Found {noise_count} noise traces.")
        return False
    else:
        print("SUCCESS: Recovered all real traces.")
        return True

if __name__ == "__main__":
    if test_mtt_initialization():
        sys.exit(0)
    else:
        sys.exit(1)
