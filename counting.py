"""자동 object counting (Homework #5 파이프라인 기반).

HW05 순서:  median filter → thresholding(Otsu) → morphology → connected component labeling
여기에 더한 것:
  1) 컬러 정보 사용 (Lab 색공간)  - 노란 잎 vs 회색 바닥처럼 밝기보다 '색'이 다른 경우
  2) 배경 보정                    - 조명 차이·바닥 무늬를 지우고 물체만 남김
  3) watershed                    - 붙어 있는 물체(동전, 레고) 나누기
  4) 자동 선택                    - 여러 설정을 모두 돌려보고 '물체다운' 결과를 고름
"""
import numpy as np
import scipy.ndimage as nd
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

WORK_SIZE = 1000   # 긴 변을 1000px로 맞춰서 처리 (사진마다 기준 통일)


# ------------------------------------------------------------------ 1. 특징 영상 만들기
def _local_median(x, win):
    """win 크기 창의 median = '그 주변의 배경'. 속도를 위해 줄여서 계산 후 다시 키움."""
    d = max(1, win // 15)
    small = x[::d, ::d]
    m = nd.median_filter(small, size=max(3, (win // d) | 1))
    m = nd.zoom(m, (x.shape[0] / m.shape[0], x.shape[1] / m.shape[1]), order=1)
    return m[: x.shape[0], : x.shape[1]]


def make_features(rgb):
    """물체가 배경과 얼마나 다른지를 나타내는 영상(값이 클수록 물체)을 여러 방식으로 만든다."""
    L = max(rgb.shape[:2])
    lab = rgb2lab(rgb)
    gray = nd.median_filter(lab[..., 0], 3)                    # 밝기 (median filter로 노이즈 제거)
    ab = nd.median_filter(lab[..., 1:], size=(3, 3, 1))         # 색 (a: 초록↔빨강, b: 파랑↔노랑)
    feats = []

    # (1) 밝기 + top-hat : 배경보다 어둡거나 밝은 '작은' 물체
    for pct in [3, 5, 8, 12, 20]:
        s = max(3, int(L * pct / 100) | 1)
        feats.append((f"밝기·어두운 물체·{pct}%", nd.black_tophat(gray, size=s)))
        feats.append((f"밝기·밝은 물체·{pct}%", nd.white_tophat(gray, size=s)))

    # (2) 주변 배경과의 색 차이 / 색+밝기 차이
    for pct in [5, 12, 25]:
        s = int(L * pct / 100)
        bg_ab = np.stack([_local_median(ab[..., i], s) for i in range(2)], -1)
        bg_g = _local_median(gray, s)
        d_ab = ((ab - bg_ab) ** 2).sum(-1)
        feats.append((f"색 차이·{pct}%", np.sqrt(d_ab)))
        feats.append((f"색+밝기 차이·{pct}%", np.sqrt(d_ab + (gray - bg_g) ** 2)))

    # (3) 사진 테두리 색(=배경색)과의 차이 : 단색 배경 위에 큰 물체가 많은 경우
    border = np.concatenate([lab[:5].reshape(-1, 3), lab[-5:].reshape(-1, 3),
                             lab[:, :5].reshape(-1, 3), lab[:, -5:].reshape(-1, 3)])
    lab_s = np.dstack([gray, ab])
    feats.append(("테두리 배경색과 차이", np.linalg.norm(lab_s - np.median(border, 0), axis=-1)))
    return feats


# ------------------------------------------------------------------ 2. 분할 (HW05 방식)
def segment(e, split):
    L = max(e.shape)

    # Otsu threshold + hysteresis (덩어리 안에 Otsu×1.5보다 강한 픽셀이 있어야 물체로 인정)
    th = threshold_otsu(e)
    lo, hi = e[e <= th], e[e > th]
    w = hi.size / e.size
    eta = w * (1 - w) * (hi.mean() - lo.mean()) ** 2 / e.var() if lo.size and hi.size else 0.0
    mask = apply_hysteresis_threshold(e, th, th * 1.5)

    # morphology: closing → hole filling → opening (HW05와 같은 순서, 크기는 사진에 비례)
    k = max(3, int(L / 200) | 1)
    st = np.ones((k, k))
    mask = nd.binary_closing(mask, st)
    mask = nd.binary_fill_holes(mask)
    mask = nd.binary_opening(mask, st)

    # connected component labeling
    lab, n = nd.label(mask)

    # watershed: 다른 것보다 1.6배 이상 큰 덩어리만, 잘록한 곳이 있으면 나눔
    if split and n:
        dist = nd.distance_transform_edt(mask)
        r = np.median(nd.maximum(dist, lab, np.arange(1, n + 1)))      # 보통 물체의 반지름
        area = nd.sum(mask, lab, np.arange(1, n + 1))
        big = np.isin(lab, np.arange(1, n + 1)[area > 1.6 * np.median(area)])
        if big.any():
            dd = nd.gaussian_filter(np.where(big, dist, 0), 1)
            markers, _ = nd.label(h_maxima(dd, max(1.5, 0.25 * r)) & big)
            ws = watershed(-dd, markers, mask=big)
            lab = np.where(big, np.where(ws > 0, ws + n, 0), lab)
    return lab, mask, eta, th


# ------------------------------------------------------------------ 3. 점수 매기기
def evaluate(lab, mask, eta):
    L = max(lab.shape)
    props = [p for p in regionprops(lab) if p.area >= (L * 0.005) ** 2]     # 너무 작은 점 제거
    if len(props) > 600:                                                     # 질감·노이즈만 잡힌 경우
        return dict(n=len(props), sol=0.0, cv=9.0, score=-9.0, cc=np.zeros((0, 2)), kept=np.zeros_like(lab))
    if props:
        med = np.median([p.area for p in props])
        props = [p for p in props if p.area > 0.2 * med]                    # HW05의 median 기준
    n = len(props)
    areas = np.array([p.area for p in props], float)
    sol = float(np.mean([p.solidity for p in props])) if n else 0.0
    cv = float(areas.std() / areas.mean()) if n else 9.0
    score = sol - cv + 0.5 * eta - (n < 3) - (mask.mean() > 0.6)
    kept = np.where(np.isin(lab, [p.label for p in props]), lab, 0)
    cc = np.array([[p.label, p.area] for p in props], float).reshape(-1, 2)
    return dict(n=n, sol=sol, cv=cv, score=float(score), cc=cc, kept=kept)


# ------------------------------------------------------------------ 4. 자동 선택
def auto_count(rgb):
    results = []
    for name, e in make_features(rgb):
        for split in (False, True):
            lab, mask, eta, th = segment(e, split)
            r = evaluate(lab, mask, eta)
            r.update(name=name + (" + watershed" if split else ""), split=split, eta=float(eta),
                     enhanced=e, mask=mask, labels=lab)
            # watershed는 확실히 좋아질 때만 쓰도록 0.1점 깎음
            r["rank"] = r["score"] - (0.1 if split else 0)
            results.append(r)
    results.sort(key=lambda r: r["rank"], reverse=True)
    best = results[0]

    # '색 차이'와 '색+밝기 차이'가 거의 같은 점수면 색+밝기 쪽 선택 (검정·흰 물체도 포함됨)
    if best["name"].startswith("색 차이"):
        twin_name = best["name"].replace("색 차이", "색+밝기 차이")
        twin = [r for r in results if r["name"] == twin_name and r["rank"] >= best["rank"] - 0.02]
        if twin:
            best = twin[0]
    return best, results
