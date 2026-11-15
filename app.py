import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import streamlit as st
from PIL import Image, ImageOps

from counting import auto_count, WORK_SIZE

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.caption("이미지를 올리면 자동으로 물체 개수를 세어줘요. "
           "(Homework #5 기반: Median filter → 배경 보정 → Otsu thresholding → Morphology → Labeling)")

uploaded = st.file_uploader("이미지 파일을 올려주세요", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("이미지를 업로드하면 결과가 바로 나와요.")
    st.stop()

# ---------------- 이미지 불러오기 (grayscale, 긴 변 1000px) ----------------
img = ImageOps.exif_transpose(Image.open(uploaded)).convert("L")
scale = WORK_SIZE / max(img.size)
img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.BILINEAR)
f = np.array(img)

with st.spinner("가장 잘 맞는 설정을 찾는 중..."):
    r, cands = auto_count(f)

cc = r["cc"]
n_obj = len(cc)

# ---------------- 결과: 숫자 ----------------
c1, c2, c3 = st.columns(3)
c1.metric("검출된 object 수", n_obj)
c2.metric("물체 밝기 (자동 판별)", "배경보다 어두움" if r["dark"] else "배경보다 밝음")
c3.metric("배경 보정 크기 (자동 선택)", f"{r['bg_size']} px")


def show(ax, im, title, cmap="gray"):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


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

# ---------------- 결과: 처리 단계 ----------------
st.subheader("처리 단계")
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
show(axes[0], f, "1. Input")
show(axes[1], r["enhanced"], "2. Background removed")
show(axes[2], r["mask"], "3. Binary mask")
show(axes[3], np.ma.masked_equal(r["kept"], 0), "4. Labeled objects", cmap="Spectral_r")
fig.tight_layout()
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
         "pixel 수": [int(v) for v in cc[:, 1]]},
        hide_index=True,
    )
else:
    st.warning("물체를 찾지 못했어요. 물체와 배경의 밝기 차이가 뚜렷한 사진으로 다시 시도해보세요.")

# ---------------- 자동 선택 과정 ----------------
with st.expander("자동 설정은 어떻게 골랐나요?"):
    st.markdown(
        "물체 밝기 2가지 × 배경 보정 크기 6가지 = **12가지 설정**을 모두 돌려보고, "
        "아래 기준으로 점수가 가장 높은 설정을 골랐어요.\n\n"
        "- **Solidity**: 물체가 꽉 찬 모양일수록 높음 (배경 조각·무늬는 낮음)\n"
        "- **크기 CV**: 물체 크기가 서로 비슷할수록 낮음\n"
        "- **Otsu η**: 물체와 배경이 뚜렷하게 나뉠수록 높음\n\n"
        "점수 = Solidity − 크기 CV + 0.5 × Otsu η"
    )
    ranked = sorted(cands, key=lambda c: c["score"], reverse=True)
    st.dataframe(
        {"순위": list(range(1, len(ranked) + 1)),
         "물체 밝기": ["어두움" if c["dark"] else "밝음" for c in ranked],
         "배경 보정 크기(px)": [c["bg_size"] for c in ranked],
         "object 수": [c["n"] for c in ranked],
         "Solidity": [round(c["sol"], 2) for c in ranked],
         "크기 CV": [round(c["cv"], 2) for c in ranked],
         "Otsu η": [round(c["eta"], 2) for c in ranked],
         "점수": [round(c["score"], 2) for c in ranked]},
        hide_index=True,
    )
