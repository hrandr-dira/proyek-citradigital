"""
app.py  —  Aplikasi Pengolahan Citra Digital
Mata Kuliah: Pengolahan Citra Digital
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from modules.basic_ops import to_grayscale, to_binary, arithmetic_operation, logical_operation
from modules.histogram import plot_histogram
from modules.convolution import apply_filter, get_kernel_display, list_filters
from modules.morphology import apply_morphology, get_se_display, list_se, list_operations

# ─── Konfigurasi halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Image Processing",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Kustom ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main { background-color: #0a0e1a; }
[data-testid="stSidebar"] { background-color: #0d1120 !important; border-right: 1px solid #1e2540; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.5px; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0d1120; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px; color: #6b7db3;
    font-family: 'Space Mono', monospace; font-size: 12px; padding: 8px 16px;
}
.stTabs [aria-selected="true"] { background: #1a2a6c !important; color: #00d4ff !important; }

.metric-box {
    background: linear-gradient(135deg, #111827, #1a2035);
    border: 1px solid #1e2d5a;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.metric-label { font-size: 11px; color: #6b7db3; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 20px; font-weight: 700; color: #00d4ff; font-family: 'Space Mono', monospace; }

.badge {
    display: inline-block; background: #1a2a6c;
    color: #7dd3fc; border-radius: 20px;
    padding: 2px 10px; font-size: 11px; font-family: 'Space Mono', monospace;
    margin-right: 4px;
}
.badge-green { background: #0d2e1a; color: #4ade80; }
.badge-yellow { background: #2d2000; color: #fbbf24; }

div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid #1e2d5a; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: PIL → numpy ─────────────────────────────────────────────────────
def pil_to_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def display_image_pair(col_left, col_right, img_orig, img_result, label_orig="Input", label_result="Hasil"):
    """Tampilkan dua gambar berdampingan."""
    with col_left:
        st.markdown(f"**{label_orig}**")
        st.image(img_orig, use_container_width=True)
    with col_right:
        st.markdown(f"**{label_result}**")
        st.image(img_result, use_container_width=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Citra Digital")
    st.markdown("---")

    st.markdown("### 📁 Input Gambar")
    uploaded_file = st.file_uploader(
        "Upload gambar (JPG/PNG/BMP)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        key="main_upload",
    )

    uploaded_file2 = st.file_uploader(
        "Upload gambar ke-2 (opsional, untuk aritmatika/logika)",
        type=["jpg", "jpeg", "png", "bmp"],
        key="second_upload",
    )

    st.markdown("---")
    st.caption("Aplikasi Pengolahan Citra Digital\nDibuat dengan Python + Streamlit")


# ─── MAIN AREA ───────────────────────────────────────────────────────────────
st.markdown("# 🔬 Pengolahan Citra Digital")
st.markdown("Aplikasi pemrosesan gambar berbasis Python · OpenCV · Streamlit")
st.markdown("---")

if uploaded_file is None:
    st.info("⬅️  Upload gambar di sidebar untuk mulai memproses.", icon="🖼️")
    st.stop()

# Load gambar utama
pil_img1 = Image.open(uploaded_file)
img1 = pil_to_np(pil_img1)

# Load gambar kedua (opsional)
img2 = None
if uploaded_file2 is not None:
    pil_img2 = Image.open(uploaded_file2)
    img2 = pil_to_np(pil_img2)

# Info gambar di sidebar
h, w, c = img1.shape
with st.sidebar:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Dimensi</div>
        <div class="metric-value">{w} × {h}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Channel</div>
        <div class="metric-value">{c} (RGB)</div>
    </div>
    """, unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab_input, tab_grayscale, tab_binary, tab_arithmetic, tab_logic, \
    tab_histogram, tab_convolution, tab_morphology = st.tabs([
    "📷 Input",
    "🌑 Grayscale",
    "⬛ Biner",
    "➕ Aritmatika",
    "🔀 Logika",
    "📊 Histogram",
    "🔲 Konvolusi",
    "🔮 Morfologi",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — INPUT GAMBAR
# ══════════════════════════════════════════════════════════════════
with tab_input:
    st.subheader("📷 Gambar Input")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(img1, caption="Gambar Asli", use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Lebar</div><div class="metric-value">{w} px</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Tinggi</div><div class="metric-value">{h} px</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Channel</div><div class="metric-value">{c}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total Piksel</div>
            <div class="metric-value">{w*h:,}</div>
        </div>
        """, unsafe_allow_html=True)

        # Download tombol
        buf = io.BytesIO()
        pil_img1.save(buf, format="PNG")
        st.download_button("⬇ Download Gambar", buf.getvalue(), "input.png", "image/png")

    if img2 is not None:
        st.subheader("📷 Gambar Input ke-2")
        st.image(img2, caption="Gambar Kedua", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — GRAYSCALE
# ══════════════════════════════════════════════════════════════════
with tab_grayscale:
    st.subheader("🌑 Konversi ke Grayscale")
    st.caption("Mengubah citra RGB menjadi grayscale menggunakan formula: Y = 0.299R + 0.587G + 0.114B")

    gray_img = to_grayscale(img1)

    col1, col2 = st.columns(2)
    display_image_pair(col1, col2, img1, gray_img, "Original (RGB)", "Hasil Grayscale")

    # Statistik piksel
    st.markdown("#### 📈 Statistik Piksel Grayscale")
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Min", int(gray_img.min())),
        ("Max", int(gray_img.max())),
        ("Mean", f"{gray_img.mean():.2f}"),
        ("Std Dev", f"{gray_img.std():.2f}"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    buf = io.BytesIO()
    np_to_pil(gray_img).save(buf, format="PNG")
    st.download_button("⬇ Download Grayscale", buf.getvalue(), "grayscale.png", "image/png")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — BINER
# ══════════════════════════════════════════════════════════════════
with tab_binary:
    st.subheader("⬛ Konversi ke Citra Biner")
    st.caption("Thresholding: piksel ≥ threshold → 255 (putih), piksel < threshold → 0 (hitam)")

    threshold_val = st.slider("🎚️ Nilai Threshold", 0, 255, 127, 1)

    binary_img = to_binary(img1, threshold_val)

    col1, col2 = st.columns(2)
    display_image_pair(col1, col2, img1, binary_img, "Original", f"Biner (T={threshold_val})")

    # Statistik
    white_pct = (binary_img == 255).sum() / binary_img.size * 100
    black_pct = 100 - white_pct
    st.markdown(f"**Piksel Putih:** {white_pct:.1f}%  &nbsp;|&nbsp;  **Piksel Hitam:** {black_pct:.1f}%")

    buf = io.BytesIO()
    np_to_pil(binary_img).save(buf, format="PNG")
    st.download_button("⬇ Download Biner", buf.getvalue(), "binary.png", "image/png")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — ARITMATIKA
# ══════════════════════════════════════════════════════════════════
with tab_arithmetic:
    st.subheader("➕ Operasi Aritmatika")
    st.caption("Operasi matematika piksel-per-piksel: penjumlahan, pengurangan, perkalian, pembagian")

    op_type = st.radio(
        "Jenis Operasi",
        ["Dua Gambar", "Gambar + Scalar"],
        horizontal=True,
    )

    if op_type == "Gambar + Scalar":
        arith_op = st.selectbox(
            "Operasi",
            ["add_scalar", "subtract_scalar"],
            format_func=lambda x: {"add_scalar": "Tambah (+)", "subtract_scalar": "Kurang (−)"}[x],
        )
        scalar_val = st.slider("Nilai Scalar", 0, 255, 50)
        result_arith = arithmetic_operation(img1, None, arith_op, scalar=scalar_val)
        col1, col2 = st.columns(2)
        display_image_pair(col1, col2, img1, result_arith,
                           "Original", f"Setelah {arith_op.replace('_', ' ')} {scalar_val}")
    else:
        if img2 is None:
            st.warning("⚠️ Upload gambar ke-2 di sidebar untuk operasi antar dua gambar.")
        else:
            arith_op2 = st.selectbox(
                "Operasi",
                ["add", "subtract", "multiply", "divide"],
                format_func=lambda x: {"add": "Penjumlahan (+)", "subtract": "Pengurangan (−)",
                                        "multiply": "Perkalian (×)", "divide": "Pembagian (÷)"}[x],
            )
            result_arith = arithmetic_operation(img1, img2, arith_op2)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Gambar 1**"); st.image(img1, use_container_width=True)
            with col2:
                st.markdown("**Gambar 2**"); st.image(img2, use_container_width=True)
            with col3:
                st.markdown("**Hasil**"); st.image(result_arith, use_container_width=True)

    if 'result_arith' in locals():
        buf = io.BytesIO()
        np_to_pil(result_arith).save(buf, format="PNG")
        st.download_button("⬇ Download Hasil Aritmatika", buf.getvalue(), "arithmetic.png", "image/png")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — LOGIKA
# ══════════════════════════════════════════════════════════════════
with tab_logic:
    st.subheader("🔀 Operasi Logika")
    st.caption("Operasi bitwise pada piksel: AND, OR, XOR, NOT")

    logic_op = st.selectbox("Pilih Operasi Logika", ["AND", "OR", "XOR", "NOT"])

    if logic_op != "NOT" and img2 is None:
        st.warning("⚠️ Upload gambar ke-2 di sidebar untuk operasi AND / OR / XOR.")
    else:
        result_logic = logical_operation(img1, img2 if img2 is not None else img1, logic_op)

        if logic_op == "NOT":
            col1, col2 = st.columns(2)
            display_image_pair(col1, col2, img1, result_logic, "Original", f"NOT (Inversi)")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Gambar 1**"); st.image(img1, use_container_width=True)
            with col2:
                st.markdown("**Gambar 2**"); st.image(img2, use_container_width=True)
            with col3:
                st.markdown(f"**Hasil {logic_op}**"); st.image(result_logic, use_container_width=True)

        buf = io.BytesIO()
        np_to_pil(result_logic).save(buf, format="PNG")
        st.download_button("⬇ Download Hasil Logika", buf.getvalue(), "logic.png", "image/png")


# ══════════════════════════════════════════════════════════════════
# TAB 6 — HISTOGRAM  (Optional, 10 poin)
# ══════════════════════════════════════════════════════════════════
with tab_histogram:
    st.subheader("📊 Histogram")
    st.caption("Distribusi intensitas piksel pada gambar input")

    hist_mode = st.radio("Mode", ["Gambar Asli (RGB)", "Gambar Grayscale"], horizontal=True)

    if hist_mode == "Gambar Grayscale":
        hist_input = to_grayscale(img1)
    else:
        hist_input = img1

    fig = plot_histogram(hist_input)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.image(hist_input, caption="Gambar yang dianalisis", use_container_width=True, width=400)


# ══════════════════════════════════════════════════════════════════
# TAB 7 — KONVOLUSI  (Optional, 20 poin)
# ══════════════════════════════════════════════════════════════════
with tab_convolution:
    st.subheader("🔲 Konvolusi & Filter")
    st.caption("Terapkan filter konvolusi pada gambar menggunakan kernel yang tersedia")

    filter_name = st.selectbox("🎛️ Pilih Filter", list_filters())

    # Tampilkan kernel
    with st.expander("🔍 Lihat Matriks Kernel"):
        kernel_np = get_kernel_display(filter_name)
        st.table(kernel_np)

    conv_mode = st.radio("Input", ["RGB", "Grayscale"], horizontal=True, key="conv_mode")
    conv_input = to_grayscale(img1) if conv_mode == "Grayscale" else img1

    result_conv = apply_filter(conv_input, filter_name)

    col1, col2 = st.columns(2)
    display_image_pair(col1, col2, conv_input, result_conv,
                       f"Input ({conv_mode})", f"Hasil: {filter_name}")

    buf = io.BytesIO()
    np_to_pil(result_conv).save(buf, format="PNG")
    st.download_button("⬇ Download Hasil Filter", buf.getvalue(), "filtered.png", "image/png")


# ══════════════════════════════════════════════════════════════════
# TAB 8 — MORFOLOGI  (Optional, 20 poin)
# ══════════════════════════════════════════════════════════════════
with tab_morphology:
    st.subheader("🔮 Operasi Morfologi")
    st.caption("Dilasi, Erosi, Opening, dan Closing menggunakan berbagai Structuring Element")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        morph_op = st.selectbox("⚙️ Operasi Morfologi", list_operations())
    with col_b:
        se_name = st.selectbox("🔷 Structuring Element", list_se())
    with col_c:
        morph_iter = st.slider("🔁 Iterasi", 1, 5, 1)

    # Tampilkan matriks SE
    with st.expander("🔍 Lihat Matriks Structuring Element"):
        se_matrix = get_se_display(se_name)
        st.table(se_matrix)

    morph_mode = st.radio("Input", ["Grayscale", "RGB"], horizontal=True, key="morph_mode")
    morph_input = to_grayscale(img1) if morph_mode == "Grayscale" else img1

    result_morph = apply_morphology(morph_input, morph_op, se_name, morph_iter)

    col1, col2 = st.columns(2)
    display_image_pair(col1, col2, morph_input, result_morph,
                       f"Input ({morph_mode})", f"Hasil: {morph_op}\n({se_name})")

    # Bandingkan semua SE untuk operasi yang dipilih
    with st.expander(f"🔀 Bandingkan Semua SE untuk Operasi '{morph_op}'"):
        se_list = list_se()
        cols = st.columns(4)
        for i, se in enumerate(se_list):
            with cols[i % 4]:
                r = apply_morphology(morph_input, morph_op, se, 1)
                st.image(r, caption=se, use_container_width=True)

    buf = io.BytesIO()
    np_to_pil(result_morph).save(buf, format="PNG")
    st.download_button("⬇ Download Hasil Morfologi", buf.getvalue(), "morphology.png", "image/png")
