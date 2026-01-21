"""
Fiber matching routines for KSPEC.

This module provides functions to match detected traces to physical fibers
for various instruments (TAIPAN, ISOPLANE).
"""

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


def taipan_nominal_fibpos(spectid: str, n: int) -> np.ndarray:
    """
    Get nominal fiber positions for TAIPAN.

    Parameters
    ----------
    spectid : str
        Spectrograph ID ('B...' for Blue, otherwise Red)
    n : int
        Number of fibers (typically 150 or 300)

    Returns
    -------
    np.ndarray
        Array of nominal positions
    """
    nomfpos = np.zeros(n, dtype=np.float32)

    if spectid.strip().upper().startswith('B'):
        _taipan_nominal_fibpos_blue(nomfpos, n)
    else:
        _taipan_nominal_fibpos_red(nomfpos, n)

    return nomfpos

def _taipan_nominal_fibpos_red(nomfpos: np.ndarray, n: int) -> None:
    """TAIPAN Red nominal positions (Fortran port)."""
    # Initialize with 0.0 or explicit values
    # Note: Fortran indices are 1-based, Python 0-based.
    # The provided list goes up to 150.

    # We use a helper dict or list to populate.
    # For brevity, I'll paste the values.
    # Note: The Fortran code has values for 1..150.
    # Python indices 0..149.

    vals = {
        1: 548., 2: 554.415344, 3: 560.507996, 4: 566.709839, 5: 573.,
        6: 579.097046, 7: 585.320740, 8: 591.528015, 9: 597.549622, 10: 603.798462,
        11: 610., 12: 616.079102, 13: 622.334167, 14: 628.491089, 15: 634.729492,
        16: 640.913086, 17: 647.033264, 18: 653.222412, 19: 659.5, 20: 665.563538,
        21: 671.694092, 22: 677.911255, 23: 684.061951, 24: 690.304565, 25: 696.443054,
        26: 702.534241, 27: 708.772766, 28: 714.884705, 29: 721., 30: 727.249146,
        31: 733.341736, 32: 739.529480, 33: 746., 34: 751.851318, 35: 758.015747,
        36: 764.200806, 37: 770.319336, 38: 776.525818, 39: 782.574280, 40: 788.812134,
        41: 794.984009, 42: 801.162476, 43: 807.354126, 44: 813., 45: 819.549316,
        46: 825.657959, 47: 831.969055, 48: 838.006287, 49: 844.259216, 50: 850.375244,
        51: 856.564392, 52: 863., 53: 868.844177, 54: 875.013428, 55: 881.087463,
        56: 887., 57: 893., 58: 899., 59: 905.714539, 60: 911.940918,
        61: 918.076904, 62: 924.302673, 63: 930., 64: 936., 65: 942.,
        66: 948., 67: 954.951172, 68: 961.064453, 69: 967.360107, 70: 973.398438,
        71: 979., 72: 985.755920, 73: 991.855408, 74: 998., 75: 1004.15778,
        76: 1030., 77: 1036.33142, 78: 1042.43713, 79: 1048.64746, 80: 1054.70422,
        81: 1061., 82: 1067., 83: 1073.21948, 84: 1079.31677, 85: 1085.,
        86: 1091., 87: 1097., 88: 1103.96448, 89: 1110.24121, 90: 1116.27808,
        91: 1122.44165, 92: 1128.58093, 93: 1134.69580, 94: 1140.87122, 95: 1147.02026,
        96: 1153.21191, 97: 1159.30164, 98: 1165.45300, 99: 1171.69958, 100: 1177.81079,
        101: 1184., 102: 1190.13293, 103: 1196.28943, 104: 1202.38770, 105: 1208.,
        106: 1214., 107: 1220., 108: 1227.18372, 109: 1233.18311, 110: 1239.37463,
        111: 1245.53088, 112: 1251., 113: 1257.85986, 114: 1263.98279, 115: 1270.17651,
        116: 1276.32129, 117: 1282., 118: 1288., 119: 1294.80042, 120: 1300.98804,
        121: 1307.11743, 122: 1313.31445, 123: 1319.41931, 124: 1325.69397, 125: 1331.74255,
        126: 1337.94006, 127: 1344.04761, 128: 1350.33093, 129: 1356.40161, 130: 1362.64661,
        131: 1368.79590, 132: 1374.89453, 133: 1381.08521, 134: 1387.26904, 135: 1393.47388,
        136: 1399.69324, 137: 1405.78491, 138: 1412., 139: 1418., 140: 1424.37390,
        141: 1430.52942, 142: 1436.69763, 143: 1442.84387, 144: 1448.96619, 145: 1455.29272,
        146: 1461.40161, 147: 1467., 148: 1473.76782, 149: 1479.96619, 150: 1486.11108
    }

    for i in range(1, 151):
        if i in vals and (i-1) < n:
            nomfpos[i-1] = vals[i]

    # Extrapolate beyond 150 if needed
    for i in range(151, n + 1):
        nomfpos[i-1] = nomfpos[i-2] + 6.20

def _taipan_nominal_fibpos_blue(nomfpos: np.ndarray, n: int) -> None:
    """TAIPAN Blue nominal positions (Fortran port)."""
    vals = {
        1: 565.343628, 2: 571.655884, 3: 577.861572, 4: 584.065796, 5: 589.641113,
        6: 595.957336, 7: 602.231201, 8: 608.371765, 9: 614.411743, 10: 620.749573,
        11: 626., 12: 632.793701, 13: 639.087036, 14: 645.208252, 15: 651.420532,
        16: 657.547607, 17: 663.671570, 18: 669.952271, 19: 676., 20: 681.933960,
        21: 688.172058, 22: 694.366150, 23: 700.513855, 24: 706.735046, 25: 712.823547,
        26: 718.915894, 27: 725.081604, 28: 731.275696, 29: 737., 30: 743.316589,
        31: 749.537659, 32: 755.790466, 33: 763., 34: 767.631348, 35: 773.999756,
        36: 780.220886, 37: 786.367615, 38: 792.487915, 39: 798.440247, 40: 804.772705,
        41: 810.909363, 42: 817.038330, 43: 823.256287, 44: 829., 45: 835.299438,
        46: 841.590454, 47: 847.676331, 48: 854., 49: 860., 50: 865.925476,
        51: 872.230347, 52: 878., 53: 884.218445, 54: 890.464966, 55: 896.668518,
        56: 903., 57: 909., 58: 915., 59: 920.936218, 60: 927.227478,
        61: 933.370605, 62: 939.640625, 63: 946., 64: 951.570618, 65: 958.,
        66: 964., 67: 970.051086, 68: 976., 69: 982.275146, 70: 988.507690,
        71: 995., 72: 1000.58307, 73: 1006.85590, 74: 1013., 75: 1018.99005,
        76: 1045., 77: 1050.81653, 78: 1057.02307, 79: 1063.32983, 80: 1069.,
        81: 1075., 82: 1081., 83: 1087.57166, 84: 1093.83447, 85: 1100.,
        86: 1106., 87: 1112., 88: 1118.17883, 89: 1124.06311, 90: 1130.43115,
        91: 1136.64307, 92: 1142.86267, 93: 1147., 94: 1154.90137, 95: 1161.09888,
        96: 1167.25354, 97: 1173.31445, 98: 1179.45752, 99: 1185.63171, 100: 1191.83582,
        101: 1198., 102: 1203.88940, 103: 1210.10315, 104: 1216.29309, 105: 1224.,
        106: 1230., 107: 1236., 108: 1240.30908, 109: 1246.75598, 110: 1252.96777,
        111: 1259.18042, 112: 1265., 113: 1271.22607, 114: 1277.44702, 115: 1283.58691,
        116: 1289.77917, 117: 1296., 118: 1302., 119: 1307.97559, 120: 1314.21069,
        121: 1320.29944, 122: 1326.47876, 123: 1332.65381, 124: 1338.69946, 125: 1344.74951,
        126: 1350.98743, 127: 1357.06653, 128: 1363.25598, 129: 1369.33936, 130: 1375.62256,
        131: 1381., 132: 1387.59753, 133: 1393.87598, 134: 1400.12061, 135: 1406.18933,
        136: 1412.31274, 137: 1418.57104, 138: 1425., 139: 1431., 140: 1436.83557,
        141: 1443.02356, 142: 1449.17566, 143: 1455.32031, 144: 1461.39673, 145: 1467.59204,
        146: 1473.84741, 147: 1480.26318, 148: 1485.60266, 149: 1492.09705, 150: 1498.36023
    }

    for i in range(1, 151):
        if i in vals and (i-1) < n:
            nomfpos[i-1] = vals[i]

    # Extrapolate beyond 150 if needed (assuming same gap)
    for i in range(151, n + 1):
        nomfpos[i-1] = nomfpos[i-2] + 6.20 # Using Red gap as fallback

def match_fibers_taipan(
    nf: int,
    fibre_types: np.ndarray,
    pk_posn: np.ndarray,
    ar_posn: np.ndarray,
    max_del: float = 4.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match traces to fibers for TAIPAN using LAP (Linear Assignment Problem).

    Parameters
    ----------
    nf : int
        Number of fibers
    fibre_types : np.ndarray
        Array of fibre types ('P', 'S', 'N', etc.)
    pk_posn : np.ndarray
        Detected peak positions
    ar_posn : np.ndarray
        Archived/Nominal fiber positions
    max_del : float
        Maximum acceptable deviation (pixels)

    Returns
    -------
    tuple
        (match_vector, modelled_positions)
        match_vector: Array of size NF, where match_vector[fib_idx] = trace_idx (1-based)
        modelled_positions: Array of size NF with fitted positions
    """
    npks = len(pk_posn)
    match_vector = np.zeros(nf, dtype=int)
    modelled_posn = ar_posn.copy()

    # Filter fibers to include (P and S types)
    include_fib_indices = []
    for i in range(nf):
        if fibre_types[i] in ['P', 'S']:
            include_fib_indices.append(i)

    m = len(include_fib_indices) # Number of fibers to match
    n_traces = npks

    if m == 0 or n_traces == 0:
        logger.warning("No fibers or traces to match")
        return match_vector, modelled_posn

    # Construct Cost Matrix (M x N)
    # Rows: Fibers (subset), Cols: Traces
    # We pad to square matrix for linear_sum_assignment if needed,
    # but scipy handles rectangular.
    # However, to handle "failed matches" (dist > max_del), we can use a large cost.

    cost_matrix = np.zeros((m, n_traces))
    max_weight = 1.0e8

    for i, fib_idx in enumerate(include_fib_indices):
        nominal_pos = ar_posn[fib_idx]
        for j in range(n_traces):
            dist = abs(nominal_pos - pk_posn[j])
            if dist < max_del:
                cost_matrix[i, j] = dist
            else:
                cost_matrix[i, j] = max_weight

    # Solve LAP
    # row_ind corresponds to index in include_fib_indices
    # col_ind corresponds to trace index
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Assign matches
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_weight:
            fib_idx = include_fib_indices[r]
            # trace indices are 1-based in kspecdr/2dfdr convention usually?
            # make_tlm.py uses 0-based indexing for arrays, but match_vector usually holds 1-based index?
            # In `convert_traces_to_tramline_map`:
            # traceno = match_vector[fibno]
            # tramline_map[:, fibno] = traces[:, traceno - 1]
            # So match_vector holds 1-based trace index.
            match_vector[fib_idx] = c + 1
            modelled_posn[fib_idx] = pk_posn[c]

    # Second Pass: Match N and U types to remaining traces
    # Identify unmatched traces
    matched_traces = set(col_ind[cost_matrix[row_ind, col_ind] < max_weight])

    for fib_idx in range(nf):
        if fibre_types[fib_idx] in ['N', 'U']:
            if match_vector[fib_idx] != 0:
                continue

            nominal_pos = ar_posn[fib_idx]
            best_trace = -1
            min_dist = max_del

            for t_idx in range(n_traces):
                if t_idx in matched_traces:
                    continue

                dist = abs(nominal_pos - pk_posn[t_idx])
                if dist < min_dist:
                    min_dist = dist
                    best_trace = t_idx

            if best_trace != -1:
                match_vector[fib_idx] = best_trace + 1
                modelled_posn[fib_idx] = pk_posn[best_trace]
                matched_traces.add(best_trace)

    return match_vector, modelled_posn

def match_fibers_isoplane(
    nf: int,
    pk_posn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 1-to-1 fiber matching for ISOPLANE.

    Parameters
    ----------
    nf : int
        Number of fibers
    pk_posn : np.ndarray
        Detected peak positions

    Returns
    -------
    tuple
        (match_vector, modelled_positions)
    """
    match_vector = np.zeros(nf, dtype=int)
    # Just use 0.0 for modelled pos if unknown, or peak pos if matched
    modelled_posn = np.zeros(nf)

    npks = len(pk_posn)

    # Assume traces correspond to fibers 1..min(NF, NPKS)
    # Or 0..min-1
    count = min(nf, npks)

    for i in range(count):
        match_vector[i] = i + 1 # 1-based trace index
        modelled_posn[i] = pk_posn[i]

    return match_vector, modelled_posn
