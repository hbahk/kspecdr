def interpolate_tlms_taipan(nx, nf, tlma, match_vector, nominal_positions):
    """
    TAIPAN 전용 파이버 보간
    - 명목 위치를 사용한 정확한 보간
    """
    
    # 최소 2개 파이버가 매칭되어야 함
    if count_nonzero(match_vector[:nf]) < 2:
        print("Warning! Too few matched peaks to interpolate with")
        return tlma
    
    # 하단 끝에서 외삽
    fibno1 = find_first_matched_fibre(match_vector, 1, nf)
    for fibno in range(fibno1-1, 0, -1):
        delta = nominal_positions[fibno1] - nominal_positions[fibno]
        tlma[:, fibno] = tlma[:, fibno1] - delta
    
    # 상단 끝에서 외삽
    fibno2 = find_last_matched_fibre(match_vector, 1, nf)
    for fibno in range(fibno2+1, nf+1):
        delta = nominal_positions[fibno] - nominal_positions[fibno2]
        tlma[:, fibno] = tlma[:, fibno2] + delta
    
    # 중간 파이버들 보간
    for fibno in range(fibno1+1, fibno2):
        if match_vector[fibno] != 0:
            continue
            
        # 위쪽과 아래쪽 매칭된 파이버 찾기
        fibno_above = find_next_matched_fibre(match_vector, fibno, nf)
        fibno_below = find_prev_matched_fibre(match_vector, fibno, 1)
        
        if fibno_above is None or fibno_below is None:
            continue
        
        # 명목 위치 기반 선형 보간
        lambda_val = (nominal_positions[fibno] - nominal_positions[fibno_below]) / \
                     (nominal_positions[fibno_above] - nominal_positions[fibno_below])
        
        tlma[:, fibno] = (1.0 - lambda_val) * tlma[:, fibno_below] + \
                         lambda_val * tlma[:, fibno_above]
    
    return tlma

def find_first_matched_fibre(match_vector, start, end):
    """첫 번째 매칭된 파이버 찾기"""
    for i in range(start, end + 1):
        if match_vector[i-1] != 0:  # 0-based indexing
            return i
    return None

def find_last_matched_fibre(match_vector, start, end):
    """마지막 매칭된 파이버 찾기"""
    for i in range(end, start - 1, -1):
        if match_vector[i-1] != 0:  # 0-based indexing
            return i
    return None

def find_next_matched_fibre(match_vector, current, max_fib):
    """현재 위치에서 다음 매칭된 파이버 찾기"""
    for i in range(current + 1, max_fib + 1):
        if match_vector[i-1] != 0:  # 0-based indexing
            return i
    return None

def find_prev_matched_fibre(match_vector, current, min_fib):
    """현재 위치에서 이전 매칭된 파이버 찾기"""
    for i in range(current - 1, min_fib - 1, -1):
        if match_vector[i-1] != 0:  # 0-based indexing
            return i
    return None

def count_nonzero(arr):
    """0이 아닌 요소 개수 세기"""
    return np.count_nonzero(arr)

def taipan_nominal_fibpos(spectid, n):
    """TAIPAN 명목 파이버 위치 반환"""
    if spectid.startswith('B'):
        return taipan_nominal_fibpos_blue(n)
    else:
        return taipan_nominal_fibpos_red(n)
