"""HW05 object counting 처리 로직 (Streamlit 화면과 분리)."""
import numpy as np
import scipy.signal as signal
import scipy.ndimage as nd
from skimage.filters import threshold_otsu


def _components(mask, size_ratio):
    """connected component labeling + 픽셀 수 + 작은 조각 제거."""
    labels, nums = nd.label(mask)
    if nums == 0:
        empty = np.zeros((0, 2))
        return labels, empty, empty, np.zeros_like(labels)
    sizes = nd.sum(mask, labels, index=np.arange(1, nums + 1))
    all_cc = np.column_stack([np.arange(1, nums + 1), sizes])
    # median의 size_ratio배 이하인 조각은 노이즈로 보고 제거
    cc = all_cc[all_cc[:, 1] > size_ratio * np.median(all_cc[:, 1])]
    kept = np.where(np.isin(labels, cc[:, 0].astype(int)), labels, 0)
    return labels, all_cc, cc, kept


def count_hw05(f, median_size=7, use_otsu=True, manual_th=149, dark_objects=True,
               low_cut=30, morph_size=9, size_ratio=0.5):
    """[방식 1] HW05와 똑같은 순서."""
    f_denoised = signal.medfilt2d(f.astype(np.float64), median_size)

    th = float(threshold_otsu(f_denoised)) if use_otsu else float(manual_th)
    if dark_objects:
        mask = (f_denoised < th) & (f_denoised > low_cut)
    else:
        mask = f_denoised > th

    st = np.ones((morph_size, morph_size))
    mask = nd.binary_closing(mask, structure=st)
    mask = nd.binary_fill_holes(mask)
    mask = nd.binary_opening(mask, structure=st)

    labels, all_cc, cc, kept = _components(mask, size_ratio)
    return dict(denoised=f_denoised, enhanced=None, th=th, mask=mask,
                labels=labels, all_cc=all_cc, cc=cc, kept=kept)


def count_tophat(f, median_size=3, dark_objects=False, bg_size=41, use_otsu=True,
                 manual_th=40, morph_size=3, size_ratio=0.2):
    """[방식 2] 배경 보정(top-hat) 후 counting.

    top-hat = 원본 - opening(원본)
    opening은 bg_size보다 작은 밝은 물체를 지워서 '배경만 남은 영상'을 만든다.
    원본에서 그 배경을 빼면 작은 물체만 남고, 조명 차이·큰 타일 같은 배경은 사라진다.
    """
    f_denoised = nd.median_filter(f.astype(np.float64), size=median_size)

    if dark_objects:
        enhanced = nd.black_tophat(f_denoised, size=bg_size)  # 어두운 작은 물체
    else:
        enhanced = nd.white_tophat(f_denoised, size=bg_size)  # 밝은 작은 물체

    th = float(threshold_otsu(enhanced)) if use_otsu else float(manual_th)
    mask = enhanced > th

    if morph_size > 1:
        mask = nd.binary_opening(mask, structure=np.ones((morph_size, morph_size)))
    mask = nd.binary_fill_holes(mask)

    labels, all_cc, cc, kept = _components(mask, size_ratio)
    return dict(denoised=f_denoised, enhanced=enhanced, th=th, mask=mask,
                labels=labels, all_cc=all_cc, cc=cc, kept=kept)
