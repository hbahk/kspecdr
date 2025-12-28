def taipan_nominal_fibpos_red(n):
    """TAIPAN Red 채널 명목 파이버 위치 (150개)"""
    positions = np.array([
        548.0, 554.415344, 560.507996, 566.709839, 573.0,
        579.097046, 585.320740, 591.528015, 597.549622, 603.798462,
        610.0, 616.079102, 622.334167, 628.491089, 634.729492,
        640.913086, 647.033264, 653.222412, 659.5, 665.563538,
        671.694092, 677.911255, 684.061951, 690.304565, 696.443054,
        702.534241, 708.772766, 714.884705, 721.0, 727.249146,
        733.341736, 739.529480, 746.0, 751.851318, 758.015747,
        764.200806, 770.319336, 776.525818, 782.574280, 788.812134,
        794.984009, 801.162476, 807.354126, 813.0, 819.549316,
        825.657959, 831.969055, 838.006287, 844.259216, 850.375244,
        856.564392, 863.0, 868.844177, 875.013428, 881.087463,
        887.0, 893.0, 899.0, 905.714539, 911.940918,
        918.076904, 924.302673, 930.0, 936.0, 942.0,
        948.0, 954.951172, 961.064453, 967.360107, 973.398438,
        979.0, 985.755920, 991.855408, 998.0, 1004.15778,
        1030.0, 1036.33142, 1042.43713, 1048.64746, 1054.70422,
        1061.0, 1067.0, 1073.21948, 1079.31677, 1085.0,
        1091.0, 1097.0, 1103.96448, 1110.24121, 1116.27808,
        1122.44165, 1128.58093, 1134.69580, 1140.87122, 1147.02026,
        1153.21191, 1159.30164, 1165.45300, 1171.69958, 1177.81079,
        1184.0, 1190.13293, 1196.28943, 1202.38770, 1208.0,
        1214.0, 1220.0, 1227.18372, 1233.18311, 1239.37463,
        1245.53088, 1251.0, 1257.85986, 1263.98279, 1270.17651,
        1276.32129, 1282.0, 1288.0, 1294.80042, 1300.98804,
        1307.11743, 1313.31445, 1319.41931, 1325.69397, 1331.74255,
        1337.94006, 1344.04761, 1350.33093, 1356.40161, 1362.64661,
        1368.79590, 1374.89453, 1381.08521, 1387.26904, 1393.47388,
        1399.69324, 1405.78491, 1412.0, 1418.0, 1424.37390,
        1430.52942, 1436.69763, 1442.84387, 1448.96619, 1455.29272,
        1461.40161, 1467.0, 1473.76782
    ])
    return positions[:n]

def taipan_nominal_fibpos_blue(n):
    """TAIPAN Blue 채널 명목 파이버 위치"""
    # Red 위치에 오프셋 적용
    red_positions = taipan_nominal_fibpos_red(n)
    blue_positions = red_positions + 12.0 + 1.8
    
    # 첫 75개 파이버에 추가 오프셋
    blue_positions[:75] += 1.5
    
    return blue_positions

def ident_fibs_from_posns(nf, fibre_types, npks, pk_posn, nominal_positions, match_vector, mfib_posn):
    """
    명목 위치를 사용한 파이버 식별
    """
    # match_vector 초기화
    match_vector.fill(0)
    
    # 각 검출된 피크에 대해 가장 가까운 명목 위치 찾기
    for pk_idx in range(npks):
        min_distance = float('inf')
        best_fibno = -1
        
        for fibno in range(nf):
            # P, S 타입 파이버만 고려
            if fibre_types[fibno] not in ['P', 'S']:
                continue
                
            distance = abs(pk_posn[pk_idx] - nominal_positions[fibno])
            if distance < min_distance:
                min_distance = distance
                best_fibno = fibno
        
        # 허용 가능한 거리 내에 있으면 매칭
        if min_distance < 5.0:  # 5픽셀 이내
            match_vector[best_fibno] = pk_idx + 1  # 1-based indexing
            mfib_posn[best_fibno] = pk_posn[pk_idx]
    
    return match_vector, mfib_posn

def convert_fibre_types_to_trace_status(instrument_code, fibre_types, nf):
    """
    파이버 타입을 추적 상태로 변환
    TRISTATE__YES: 확실히 흔적 있음
    TRISTATE__NO: 확실히 흔적 없음  
    TRISTATE__MAYBE: 흔적 있을 수도 있음
    """
    fibre_has_trace = np.zeros(nf, dtype=int)
    
    for i in range(nf):
        fib_type = fibre_types[i]
        
        if fib_type in ['P', 'S']:  # Program, Sky
            fibre_has_trace[i] = 1  # TRISTATE__YES
        elif fib_type in ['F', 'D']:  # Fiducial, Dead
            fibre_has_trace[i] = 0  # TRISTATE__NO
        else:  # N, U 등
            fibre_has_trace[i] = 2  # TRISTATE__MAYBE
    
    return fibre_has_trace

def count_fibres_with_trace(fibre_has_trace, trace_state):
    """특정 추적 상태의 파이버 수 세기"""
    if trace_state == 'YES':
        return np.sum(fibre_has_trace == 1)
    elif trace_state == 'NO':
        return np.sum(fibre_has_trace == 0)
    elif trace_state == 'MAYBE':
        return np.sum(fibre_has_trace == 2)
    else:
        return 0
