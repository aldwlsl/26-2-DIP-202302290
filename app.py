import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import streamlit as st
from PIL import Image, ImageOps
from skimage.segmentation import find_boundaries

from counting import auto_count, WORK_SIZE

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.caption("이미지를 올리면 자동으로 물체 개수를 세어줘요. "
           "(Homework #5 기반: Median filter → 배경 보정 → Otsu thresholding → Morphology → Labeling)")

uploaded = st.file_uploader("이미지 파일을 올려주세요", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("이미지를 업로드하면 결과가 바로 나와요.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=20)
def load_and_count(data: bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    s = WORK_SIZE / max(img.size)
    img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BILINEAR)
    rgb = np.array(img)
    best, results = auto_count(rgb)
    table = [dict(name=r["name"], n=r["n"], sol=r["sol"], cv=r["cv"], eta=r["eta"], score=r["score"])
             for r in results]
    best = {k: best[k] for k in ("name", "n", "cc", "kept", "labels", "enhanced", "mask", "score")}
    return rgb, best, table


with st.spinner("여러 설정을 비교해서 가장 잘 맞는 방법을 찾는 중... (10~20초)"):
    rgb, r, table = load_and_count(uploaded.getvalue())

cc = r["cc"]
n_obj = len(cc)

# ---------------- 결과: 숫자 ----------------
c1, c2 = st.columns([1, 2])
c1.metric("검출된 object 수", n_obj)
c2.metric("자동으로 고른 방법", r["name"])
if r["score"] < 0:
    st.warning("물체끼리 많이 붙어 있거나 배경이 복잡해서 결과가 정확하지 않을 수 있어요.")


def show(ax, im, title, cmap=None):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


# ---------------- 결과: 번호 붙인 object ----------------
st.subheader("검출 결과")
fig, ax = plt.subplots(figsize=(10, 10 * rgb.shape[0] / rgb.shape[1]))
ax.imshow(rgb)
edge = find_boundaries(r["kept"], mode="outer")
overlay = np.zeros((*edge.shape, 4))
overlay[edge] = [1, 0, 0, 1]
ax.imshow(overlay)
if n_obj > 0:
    labs = cc[:, 0].astype(int)
    centers = nd.center_of_mass(r["kept"] > 0, r["labels"], labs)
    fs = 12 if n_obj <= 30 else 8 if n_obj <= 100 else 6
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
show(axes[0], rgb, "1. Input")
show(axes[1], r["enhanced"], "2. Difference from background", cmap="gray")
show(axes[2], r["mask"], "3. Binary mask", cmap="gray")
show(axes[3], np.ma.masked_equal(r["kept"], 0) % 20, "4. Labeled objects", cmap="tab20")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------- 결과: 가장 큰 / 작은 object ----------------
if n_obj > 0:
    lab_max = int(cc[np.argmax(cc[:, 1]), 0])
    lab_min = int(cc[np.argmin(cc[:, 1]), 0])
    st.subheader("가장 큰 / 작은 object")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    show(axes[0], rgb, "input image")
    show(axes[1], r["labels"] == lab_max, f"the largest ({int(cc[:, 1].max())} px)", cmap="Spectral_r")
    show(axes[2], r["labels"] == lab_min, f"the smallest ({int(cc[:, 1].min())} px)", cmap="Spectral_r")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Object별 픽셀 수")
    st.dataframe({"object 번호": list(range(1, n_obj + 1)),
                  "pixel 수": [int(v) for v in cc[:, 1]]}, hide_index=True)
else:
    st.warning("물체를 찾지 못했어요. 물체와 배경의 차이가 뚜렷한 사진으로 다시 시도해보세요.")

# ---------------- 자동 선택 과정 ----------------
with st.expander("자동 설정은 어떻게 골랐나요?"):
    st.markdown(
        "**1. 배경과 다른 정도를 여러 방식으로 계산** (총 17가지)\n"
        "- 밝기 차이 + top-hat (어두운/밝은 물체, 배경 보정 크기 5가지)\n"
        "- 주변 배경과의 **색 차이**, **색+밝기 차이** (3가지 크기)\n"
        "- 사진 테두리 색(배경색)과의 차이\n\n"
        "**2. 각각 HW05 방식으로 분할**: Otsu threshold → closing → hole filling → opening → labeling "
        "(+ 붙은 물체를 나누는 watershed 적용 / 미적용)\n\n"
        "**3. 34개 결과에 점수를 매겨 가장 '물체다운' 결과 선택**\n"
        "- **Solidity**: 물체가 꽉 찬 모양일수록 높음 (바닥 무늬·배경 조각은 낮음)\n"
        "- **크기 CV**: 물체 크기가 서로 비슷할수록 낮음\n"
        "- **Otsu η**: 물체와 배경이 뚜렷하게 나뉠수록 높음\n\n"
        "점수 = Solidity − 크기 CV + 0.5 × Otsu η  (watershed는 0.1점 감점)"
    )
    ranked = sorted(table, key=lambda c: c["score"], reverse=True)
    st.dataframe(
        {"방법": [c["name"] for c in ranked],
         "object 수": [c["n"] for c in ranked],
         "Solidity": [round(c["sol"], 2) for c in ranked],
         "크기 CV": [round(c["cv"], 2) for c in ranked],
         "Otsu η": [round(c["eta"], 2) for c in ranked],
         "점수": [round(c["score"], 2) for c in ranked]},
        hide_index=True,
    )
