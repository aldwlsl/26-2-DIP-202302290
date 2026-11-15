"""자동 object counting (HW05 파이프라인 기반).

HW05 순서: median filter → Otsu threshold → morphology → connected component labeling
여기에 '배경 보정(top-hat)'을 더하고, 사진마다 맞는 설정을 자동으로 고른다.

[자동 선택 아이디어]
  물체 밝기(어두움/밝음) × 배경 보정 크기(6가지) = 12가지 후보를 전부 돌려보고,
  "물체다운 결과"가 나온 후보를 고른다.
    - solidity(꽉 찬 정도)가 높을수록 좋음   → 배경 조각·무늬는 울퉁불퉁해서 낮음
    - 크기가 서로 비슷할수록 좋음 (CV 낮음)  → 같은 종류 물체를 센다는 가정
    - Otsu로 잘 나뉠수록 좋음 (eta 높음)     → 물체와 배경이 뚜렷하게 구분됨
"""
import numpy as np
import scipy.ndimage as nd
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.measure import regionprops

WORK_SIZE = 1000                     # 긴 변을 1000px로 맞춰서 처리 (사진마다 기준 통일)
BG_PCTS = [3, 5, 8, 12, 20, 35]      # 배경 보정 크기 후보 (긴 변 대비 %)


def _run(fd, dark, pct):
    """후보 하나 실행: top-hat → Otsu(hysteresis) → opening/fill → labeling → 작은 조각 제거."""
    L = max(fd.shape)
    bg_size = max(3, int(L * pct / 100) | 1)

    # 1) 배경 보정 (top-hat): bg_size보다 작은 물체만 남기고 배경·조명 차이 제거
    enhanced = nd.black_tophat(fd, size=bg_size) if dark else nd.white_tophat(fd, size=bg_size)

    # 2) Otsu threshold + hysteresis (덩어리 안에 아주 밝은 픽셀이 있어야 물체로 인정)
    th = float(threshold_otsu(enhanced))
    lo, hi = enhanced[enhanced <= th], enhanced[enhanced > th]
    w = hi.size / enhanced.size
    eta = (w * (1 - w) * (hi.mean() - lo.mean()) ** 2 / enhanced.var()) if lo.size and hi.size else 0.0
    mask = apply_hysteresis_threshold(enhanced, th, th * 1.5)

    # 3) morphology: opening(가는 무늬·노이즈 제거) → hole filling
    k = max(3, int(L / 200) | 1)
    mask = nd.binary_opening(mask, structure=np.ones((k, k)))
    mask = nd.binary_fill_holes(mask)

    # 4) connected component labeling
    labels, nums = nd.label(mask)
    props = [p for p in regionprops(labels) if p.area >= (L * 0.005) ** 2]   # 너무 작은 점 제거
    if props:
        med = np.median([p.area for p in props])
        props = [p for p in props if p.area > 0.2 * med]                    # HW05의 median 기준

    n = len(props)
    areas = np.array([p.area for p in props], dtype=float)
    sol = float(np.mean([p.solidity for p in props])) if n else 0.0
    cv = float(areas.std() / areas.mean()) if n else 9.0
    score = sol - cv + 0.5 * eta - (n < 3) - (mask.mean() > 0.6)

    keep_labels = [p.label for p in props]
    kept = np.where(np.isin(labels, keep_labels), labels, 0)
    cc = np.array([[p.label, p.area] for p in props], dtype=float).reshape(-1, 2)
    return dict(dark=dark, pct=pct, bg_size=bg_size, enhanced=enhanced, th=th, mask=mask,
                labels=labels, kept=kept, cc=cc, n=n, sol=sol, cv=cv, eta=eta, score=score)


def auto_count(f):
    """grayscale 배열 f를 받아 12개 후보 중 점수가 가장 높은 결과를 돌려준다."""
    fd = nd.median_filter(f.astype(np.float64), size=3)   # 노이즈 제거 (HW05의 median filter)
    results = [_run(fd, dark, pct) for dark in (True, False) for pct in BG_PCTS]
    best = max(results, key=lambda r: r["score"])
    best["denoised"] = fd
    return best, results
