import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import streamlit as st
from PIL import Image, ImageOps

from counting import count_objects

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.caption("Homework #5 기반: Median filter → Otsu thresholding → Morphology → Connected component labeling")

# ---------------- 사이드바: 파라미터 ----------------
with st.sidebar:
    st.header("파라미터")
    median_size = st.select_slider("Median filter 크기", options=[3, 5, 7, 9, 11, 13, 15], value=7)
    use_otsu = st.checkbox("Otsu 자동 임계값 사용", value=True)
    manual_th = st.slider("수동 임계값", 0, 255, 149, disabled=use_otsu)
    obj_type = st.radio("물체 밝기", ["배경보다 어두움", "배경보다 밝음"])
    dark_objects = obj_type == "배경보다 어두움"
    low_cut = st.slider("너무 어두운 픽셀 제외 (하한)", 0, 100, 30, disabled=not dark_objects)
    morph_size = st.slider("Morphology 구조요소 크기", 1, 31, 9, step=2)
    size_ratio = st.slider("작은 조각 제거 기준 (median 대비 비율)", 0.0, 1.0, 0.5, step=0.05)
    max_side = st.selectbox("처리 해상도 (긴 변 최대 픽셀)", ["원본", 2000, 1500, 1000], index=0,
                            help="사진이 크면 처리가 느릴 수 있어요. 줄이면 빨라지지만 픽셀 수는 달라져요.")

uploaded = st.file_uploader("이미지 파일을 올려주세요", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is None:
    st.info("왼쪽에서 파라미터를 조절하고 위에 이미지를 업로드 업로드하면 결과 나옴")
    st.stop()

# ---------------- 이미지 불러오기 (grayscale) ----------------
img = ImageOps.exif_transpose(Image.open(uploaded)).convert("L")
if max_side != "원본" and max(img.size) > max_side:
    img.thumbnail((max_side, max_side))
f = np.array(img)

with st.spinner("처리 중..."):
    r = count_objects(f, median_size, use_otsu, manual_th, dark_objects,
                      low_cut, morph_size, size_ratio)

cc = r["cc"]
n_obj = len(cc)

# ---------------- 결과: 숫자 ----------------
c1, c2, c3 = st.columns(3)
c1.metric("검출된 object 수", n_obj)
c2.metric("사용한 임계값", f"{r['th']:.0f}")
c3.metric("이미지 크기", f"{f.shape[1]} × {f.shape[0]}")


def show(ax, im, title, cmap="gray"):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


# ---------------- 결과: 처리 단계 ----------------
st.subheader("처리 단계")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
show(axes[0], f, "1. Input (grayscale)")
show(axes[1], r["denoised"], "2. Median filtered")
show(axes[2], r["mask"], "3. Binary mask")
show(axes[3], np.ma.masked_equal(r["kept"], 0), "4. Labeled objects", cmap="Spectral_r")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------- 결과: 번호 붙인 object ----------------
st.subheader("검출 결과")
fig, ax = plt.subplots(figsize=(10, 10 * f.shape[0] / f.shape[1]))
ax.imshow(f, cmap="gray")
ax.imshow(np.ma.masked_equal(r["kept"], 0), cmap="Spectral_r", alpha=0.45)
for k, lab in enumerate(cc[:, 0].astype(int), start=1):
    y, x = nd.center_of_mass(r["labels"] == lab)
    ax.text(x, y, str(k), color="white", fontsize=14, ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"))
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
         "label": cc[:, 0].astype(int),
         "pixel 수": cc[:, 1].astype(int)},
        hide_index=True,
    )
else:
    st.warning("검출된 object가 없음 왼쪽 파라미터를 바꾸세요")

with st.expander("제거되기 전 전체 connected component 보기"):
    st.write(f"전체 component 수: {len(r['all_cc'])}  →  작은 조각 제거 후: {n_obj}")
    st.dataframe({"label": r["all_cc"][:, 0].astype(int),
                  "pixel 수": r["all_cc"][:, 1].astype(int)}, hide_index=True)
