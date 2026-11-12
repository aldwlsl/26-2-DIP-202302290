"""HW05 object counting 처리 로직 (Streamlit 화면과 분리)."""
import numpy as np
import scipy.signal as signal
import scipy.ndimage as nd
from skimage.filters import threshold_otsu


def count_objects(
    f,
    median_size=7,
    use_otsu=True,
    manual_th=149,
    dark_objects=True,
    low_cut=30,
    morph_size=9,
    size_ratio=0.5,
):
    """grayscale 배열 f(0~255)를 받아 HW05와 같은 순서로 object counting."""
    # 1) 노이즈 제거: median filter
    f_denoised = signal.medfilt2d(f.astype(np.float64), median_size)

    # 2) thresholding → binary mask
    th = float(threshold_otsu(f_denoised)) if use_otsu else float(manual_th)
    if dark_objects:
        mask = (f_denoised < th) & (f_denoised > low_cut)
    else:
        mask = f_denoised > th

    # 3) morphology: closing → hole filling → opening
    st = np.ones((morph_size, morph_size))
    mask = nd.binary_closing(mask, structure=st)
    mask = nd.binary_fill_holes(mask)
    mask = nd.binary_opening(mask, structure=st)

    # 4) connected component labeling + 픽셀 수
    labels, nums = nd.label(mask)
    if nums == 0:
        return dict(denoised=f_denoised, th=th, mask=mask, labels=labels,
                    all_cc=np.zeros((0, 2)), cc=np.zeros((0, 2)), kept=labels * 0)

    sizes = nd.sum(np.ones_like(labels), labels, index=np.arange(1, nums + 1))
    all_cc = np.column_stack([np.arange(1, nums + 1), sizes])

    # 5) 너무 작은 조각 제거: median의 size_ratio배 이하
    cc = all_cc[all_cc[:, 1] > size_ratio * np.median(all_cc[:, 1])]

    kept = np.where(np.isin(labels, cc[:, 0].astype(int)), labels, 0)
    return dict(denoised=f_denoised, th=th, mask=mask, labels=labels,
                all_cc=all_cc, cc=cc, kept=kept)
