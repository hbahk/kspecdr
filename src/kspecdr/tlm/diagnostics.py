def analyze_matching_results(match_vector, nf, fibre_types):
    """파이버 매칭 결과 분석"""
    n_matched = np.count_nonzero(match_vector[:nf])
    n_p_type = np.sum([1 for i in range(nf) if fibre_types[i] == 'P' and match_vector[i] != 0])
    n_s_type = np.sum([1 for i in range(nf) if fibre_types[i] == 'S' and match_vector[i] != 0])
    
    print(f"Total matched fibres: {n_matched}")
    print(f"P-type fibres matched: {n_p_type}")
    print(f"S-type fibres matched: {n_s_type}")
    
    return n_matched, n_p_type, n_s_type

def write_fibre_positions_to_file(nf, fibre_positions, filename='FIBPOS.DAT'):
    """파이버 위치를 외부 파일로 저장"""
    with open(filename, 'w') as f:
        f.write("#FIBS\n")
        for i in range(nf):
            f.write(f"{i+1} {fibre_positions[i]+0.5}\n")
    
    print(f"Fibre matched positions written to file {filename}")
