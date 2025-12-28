def get_taipan_config():
    """TAIPAN 기기 전용 설정 반환"""
    return {
        'order': 2,  # 2차 다항식
        'pk_search_method': 2,  # Wavelet Convolution
        'do_distortion': True,  # 왜곡 모델링 수행
        'n_fibres': 150,  # 실제 파이버 수 (300이 아닌 150)
        'nominal_separation': 6.0  # 명목 파이버 간격
    }

def set_instrument_specific_params(instrument_code, args):
    """기기별 특화 파라미터 설정"""
    if instrument_code == 'TAIPAN':
        config = get_taipan_config()
        order = config['order']
        pk_search_method = config['pk_search_method']
        do_distortion = config['do_distortion']

        # 추가 인자들 설정
        sparse_fibs = get_argument(args, 'SPARSE_FIBS', False)
        experimental = get_argument(args, 'TLM_FIT_RES', False)
        qad_pksearch = get_argument(args, 'QAD_PKSEARCH', False)

        return order, pk_search_method, do_distortion, sparse_fibs, experimental, qad_pksearch
