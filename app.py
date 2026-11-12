import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import streamlit as st
from PIL import Image, ImageOps

from counting import count_hw05, count_tophat

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.caption("Homework #5 기반 object counting 웹사이트")

MODE_HW = "방식 1: HW05 기본 (큰 물체, 고른 배경)"
MODE_TH = "방식 2: 배경 보정 (작은 물체, 조명 불균일)"

uploaded = st.file_uploader("이미지 파일을 올려주세요", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

with st.sidebar:
    st.header("처리 방식")
    mode = st.radio("방식 선택", [MODE_HW, MODE_TH], label_visibility="collapsed")
    obj_type = st.radio("물체 밝기", ["배경보다 어두움", "배경보다 밝음"],
                        index=0 if mode == MODE_HW else 1)
    dark_objects = obj_type == "배경보다 어두움"
    max_side = st.selectbox("처리 해상도 (긴 변 최대 픽셀)", ["원본", 2000, 1500, 1000],
                            index=0 if mode == MODE_HW else 3,
                            help="줄이면 빨라지지만 픽셀 수는 달라져요.")

if uploaded is None:
    st.info("이미지를 업로드하면 결과가 나와요. 결과가 이상하면 왼쪽에서 처리 방식을 바꿔보세요.")
    st.stop()

# ---------------- 이미지 불러오기 (grayscale) ----------------
img = ImageOps.exif_transpose(Image.open(uploaded)).convert("L")
if max_side != "원본" and max(img.size) > max_side:
    img.thumbnail((max_side, max_side))
f = np.array(img)
L = max(f.shape)

# ---------------- 세부 파라미터 ----------------
with st.sidebar:
    st.header("세부 파라미터")
    use_otsu = st.checkbox("Otsu 자동 임계값 사용", value=True)
    if mode == MODE_HW:
        median_size = st.select_slider("Median filter 크기", [3, 5, 7, 9, 11, 13, 15], value=7)
        manual_th = st.slider("수동 임계값", 0, 255, 149, disabled=use_otsu)
        low_cut = st.slider("너무 어두운 픽셀 제외 (하한)", 0, 100, 30, disabled=not dark_objects)
        morph_size = st.slider("Morphology 구조요소 크기", 1, 31, 9, step=2)
        size_ratio = st.slider("작은 조각 제거 기준 (median 대비)", 0.0, 1.0, 0.5, step=0.05)
    else:
        # 사진 크기에 맞춰 기본값 자동 설정
        bg_pct = st.slider("배경 보정 크기 (긴 변 대비 %)", 1.0, 20.0, 5.0, step=0.5,
                           help="세려는 물체보다 크고, 배경 무늬(타일 등)보다는 작게")
        bg_size = max(3, int(L * bg_pct / 100) | 1)
        median_size = st.select_slider("Median filter 크기", [1, 3, 5, 7, 9], value=3)
        manual_th = st.slider("수동 임계값", 0, 255, 40, disabled=use_otsu)
        morph_default = max(1, int(L / 300) | 1)
        morph_size = st.slider("Opening 구조요소 크기", 1, 15, min(15, morph_default), step=2)
        size_ratio = st.slider("작은 조각 제거 기준 (median 대비)", 0.0, 1.0, 0.2, step=0.05)
        st.caption(f"배경 보정 크기 = {bg_size} px")

with st.spinner("처리 중..."):
    if mode == MODE_HW:
        r = count_hw05(f, median_size, use_otsu, manual_th, dark_objects,
                       low_cut, morph_size, size_ratio)
    else:
        r = count_tophat(f, median_size, dark_objects, bg_size, use_otsu,
                         manual_th, morph_size, size_ratio)

cc = r["cc"]
n_obj = len(cc)
mask_ratio = r["mask"].mean()

# ---------------- 결과: 숫자 ----------------
c1, c2, c3 = st.columns(3)
c1.metric("검출된 object 수", n_obj)
c2.metric("사용한 임계값", f"{r['th']:.1f}")
c3.metric("이미지 크기", f"{f.shape[1]} × {f.shape[0]}")

if mask_ratio > 0.35:
    other = "방식 2(배경 보정)" if mode == MODE_HW else "방식 1(HW05 기본)"
    st.warning(f"Binary mask가 이미지의 {mask_ratio*100:.0f}%를 차지해요. 배경이 물체로 잡힌 것 같아요. "
               f"**물체 밝기**를 바꾸거나 **{other}**를 써보세요.")


def show(ax, im, title, cmap="gray"):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


# ---------------- 결과: 처리 단계 ----------------
st.subheader("처리 단계")
if r["enhanced"] is None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    show(axes[0], f, "1. Input (grayscale)")
    show(axes[1], r["denoised"], "2. Median filtered")
    show(axes[2], r["mask"], "3. Binary mask")
    show(axes[3], np.ma.masked_equal(r["kept"], 0), "4. Labeled objects", cmap="Spectral_r")
else:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    show(axes[0], f, "1. Input (grayscale)")
    show(axes[1], r["enhanced"], "2. Background removed (top-hat)")
    show(axes[2], r["mask"], "3. Binary mask")
    show(axes[3], np.ma.masked_equal(r["kept"], 0), "4. Labeled objects", cmap="Spectral_r")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------- 결과: 번호 붙인 object ----------------
st.subheader("검출 결과")
fig, ax = plt.subplots(figsize=(10, 10 * f.shape[0] / f.shape[1]))
ax.imshow(f, cmap="gray")
ax.imshow(np.ma.masked_equal(r["kept"], 0), cmap="Spectral_r", alpha=0.6)
if n_obj > 0:
    labs = cc[:, 0].astype(int)
    centers = nd.center_of_mass(r["kept"] > 0, r["labels"], labs)
    fs = 14 if n_obj <= 30 else 9 if n_obj <= 100 else 6
    for k, (y, x) in enumerate(centers, start=1):
        ax.text(x, y, str(k), color="white", fontsize=fs, ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.15"))
ax.set_title(f"Object count = {n_obj}")
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

# ---------------- 결과: 가장 큰 / 작은 object ----------------
if n_obj > 0:
    lab_max = int(cc[np.argmax(cc[:, 1]), 0])
    lab_min = int(cc[np.argmin(cc[:, 1]), 0])
    st.subheader("가장 큰 / 작은 object")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    show(axes[0], f, "input image")
    show(axes[1], r["labels"] == lab_max, f"the largest ({int(cc[:, 1].max())} px)", cmap="Spectral_r")
    show(axes[2], r["labels"] == lab_min, f"the smallest ({int(cc[:, 1].min())} px)", cmap="Spectral_r")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Object별 픽셀 수")
    st.dataframe(
        {"object 번호": list(range(1, n_obj + 1)),
         "label": [int(v) for v in cc[:, 0]],
         "pixel 수": [int(v) for v in cc[:, 1]]},
        hide_index=True,
    )
else:
    st.warning("검출된 object가 없어요. 왼쪽에서 처리 방식이나 물체 밝기를 바꿔보세요.")

with st.expander("제거되기 전 전체 connected component 보기"):
    st.write(f"전체 component 수: {len(r['all_cc'])}  →  작은 조각 제거 후: {n_obj}")
    st.dataframe({"label": [int(v) for v in r["all_cc"][:, 0]],
                  "pixel 수": [int(v) for v in r["all_cc"][:, 1]]}, hide_index=True)
