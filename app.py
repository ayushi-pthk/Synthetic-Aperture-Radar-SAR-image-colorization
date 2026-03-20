import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
from skimage.color import lab2rgb
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import io
import gdown
import os

from model import EnsembleEncoder, Decoder

# -------------------------------------------------------
# Download models if not present
# -------------------------------------------------------

ENCODER_PATH = "encoder.pth"
DECODER_PATH = "decoder.pth"

ENCODER_URL = "https://drive.google.com/uc?id=1FM0NPT8k_7Evamd80TkL20MninwWKRjs"
DECODER_URL = "https://drive.google.com/uc?id=15WwMntcv63qQmsESpDiV8Fqgxf3coy-p"


@st.cache_resource
def download_models():
    if not os.path.exists(ENCODER_PATH):
        gdown.download(ENCODER_URL, ENCODER_PATH, quiet=False)

    if not os.path.exists(DECODER_PATH):
        gdown.download(DECODER_URL, DECODER_PATH, quiet=False)


download_models()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SAR -> Color", page_icon="🛰️", layout="wide")
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
 
:root {
    --bg: #0d1117; --surface: #161b22; --surface2: #1c2330;
    --border: #30363d; --accent: #58a6ff; --accent2: #3fb950;
    --warn: #f78166; --text: #e6edf3; --muted: #8b949e;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }
 
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1b2a 100%);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(88,166,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: var(--accent); margin: 0 0 0.4rem 0; letter-spacing: -1px; }
.hero-sub   { color: var(--muted); font-size: 1rem; font-weight: 300; margin: 0; }
 
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
.card-title { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
 
.metrics-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card { flex: 1; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1.5rem; text-align: center; }
.metric-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.5rem; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-good { color: var(--accent2); } .metric-mid { color: #e3b341; } .metric-low { color: var(--warn); }
.metric-note { font-size: 0.72rem; color: var(--muted); margin-top: 0.4rem; }
 
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-good { background: rgba(63,185,80,0.15);  color: var(--accent2); }
.badge-mid  { background: rgba(227,179,65,0.15); color: #e3b341; }
.badge-low  { background: rgba(247,129,102,0.15); color: var(--warn); }
 
.compare-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.5rem; }
.compare-table th { font-family: 'Space Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
.compare-table td { padding: 0.75rem 1rem; border-bottom: 1px solid #21262d; vertical-align: middle; }
.compare-table tr:last-child td { border-bottom: none; }
 
.img-label { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; text-align: center; margin-top: 0.5rem; }
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
 
.stFileUploader > div { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
.stButton > button { background: var(--accent) !important; color: #0d1117 !important; border: none !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important; font-weight: 700 !important; padding: 0.6rem 1.5rem !important; }
.stDownloadButton > button { background: var(--surface2) !important; color: var(--accent) !important; border: 1px solid var(--accent) !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)
 
# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = EnsembleEncoder().to(device)
    dec = Decoder().to(device)
    enc.load_state_dict(torch.load("encoder.pth", map_location=device))
    dec.load_state_dict(torch.load("decoder.pth", map_location=device))
    enc.eval(); dec.eval()
    return enc, dec, device
 
encoder, decoder, device = load_models()
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_sar(img_pil):

    img_pil = img_pil.convert("RGB")   # ensure 3 channels

    t = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return t(img_pil).unsqueeze(0).to(device)
 
def run_inference(tensor):
    with torch.no_grad():
        f56, f28, f14, f7 = encoder(tensor)
        ab  = decoder(f7, f14, f28, f56)
        L   = (tensor[:, 0:1] + 1) * 0.5 * 100
        ab  = ((ab + 1) * 0.5 * 255) - 128
        lab = torch.cat([L, ab], dim=1).squeeze().cpu().permute(1, 2, 0).numpy()
    return lab2rgb(lab)
 
def compute_metrics(real_rgb, pred_rgb):
    real = (np.clip(real_rgb, 0, 1) * 255).astype(np.uint8)
    pred = (np.clip(pred_rgb, 0, 1) * 255).astype(np.uint8)
    return psnr(real, pred, data_range=255), ssim(real, pred, channel_axis=2, data_range=255)
 
def badge_psnr(v):
    return ("badge-good", "Excellent") if v >= 28 else (("badge-mid", "Good") if v >= 22 else ("badge-low", "Low"))
 
def badge_ssim(v):
    return ("badge-good", "Excellent") if v >= 0.85 else (("badge-mid", "Good") if v >= 0.65 else ("badge-low", "Low"))
 
def mc(label, v):
    if label == "PSNR":
        return "metric-good" if v >= 28 else ("metric-mid" if v >= 22 else "metric-low")
    return "metric-good" if v >= 0.85 else ("metric-mid" if v >= 0.65 else "metric-low")
 
def to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
 
def channel_row(name, real_arr, pred_arr, idx):
    """Build one table row as a plain concatenated string — avoids nested f-string issues."""
    rm  = real_arr[:, :, idx].mean()
    pm  = pred_arr[:, :, idx].mean()
    rs  = real_arr[:, :, idx].std()
    ps  = pred_arr[:, :, idx].std()
    d   = abs(rm - pm)
    cls = "badge-good" if d < 10 else ("badge-mid" if d < 25 else "badge-low")
    return (
        "<tr>"
        "<td><strong>" + name + "</strong></td>"
        "<td>" + f"{rm:.1f}" + "</td>"
        "<td>" + f"{pm:.1f}" + "</td>"
        "<td>" + f"{rs:.1f}" + "</td>"
        "<td>" + f"{ps:.1f}" + "</td>"
        "<td><span class='badge " + cls + "'>&#916; " + f"{d:.1f}" + "</span></td>"
        "</tr>"
    )
 
# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title" style="font-size: 2rem; text-align: center;">&#128752; SAR Image Colorization</p>
  <p class="hero-sub" style="text-align: center;">SAR &#8594; Optical Image Colorization &nbsp;&middot;&nbsp; Deep Learning &nbsp;&middot;&nbsp; Sentinel-1/2</p>
</div>""", unsafe_allow_html=True)
 
# ── Upload ────────────────────────────────────────────────────────────────────
col_u1, col_u2 = st.columns(2)
with col_u1:
    st.markdown('<div class="card"><div class="card-title">&#128225; Input &#8212; SAR Image</div>', unsafe_allow_html=True)
    sar_file = st.file_uploader("Upload SAR image (s1)", type=["png", "jpg", "jpeg"], key="sar")
    st.markdown('</div>', unsafe_allow_html=True)
with col_u2:
    st.markdown('<div class="card"><div class="card-title">&#127757; Reference &#8212; Real Optical Image (optional)</div>', unsafe_allow_html=True)
    opt_file = st.file_uploader("Upload matching optical image (s2) for comparison", type=["png", "jpg", "jpeg"], key="opt")
    st.markdown('</div>', unsafe_allow_html=True)
 
# ── Main ──────────────────────────────────────────────────────────────────────
if sar_file is not None:
    sar_pil  = Image.open(sar_file).convert("L")
    pred_rgb = run_inference(preprocess_sar(sar_pil))
    pred_pil = Image.fromarray((np.clip(pred_rgb, 0, 1) * 255).astype(np.uint8))
 
    has_real = opt_file is not None
    real_rgb, real_pil = None, None
    if has_real:
        real_pil = Image.open(opt_file).convert("RGB").resize((224, 224))
        real_rgb = np.array(real_pil).astype(np.float32) / 255.0
 
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
 
    # ── Metric cards ──────────────────────────────────────────────────────────
    if has_real:
        psnr_v, ssim_v = compute_metrics(real_rgb, pred_rgb)
        pb_cls, pb_lbl = badge_psnr(psnr_v)
        sb_cls, sb_lbl = badge_ssim(ssim_v)
 
        st.markdown(
            "<div class='metrics-row'>"
            "<div class='metric-card'>"
            "<div class='metric-label'>PSNR</div>"
            "<div class='metric-value " + mc("PSNR", psnr_v) + "'>" + f"{psnr_v:.2f}" + " <span style='font-size:1rem'>dB</span></div>"
            "<div class='metric-note'>&#8805;28 dB = Excellent &nbsp;|&nbsp; 22&#8211;28 = Good &nbsp;|&nbsp; &lt;22 = Low</div>"
            "</div>"
            "<div class='metric-card'>"
            "<div class='metric-label'>SSIM</div>"
            "<div class='metric-value " + mc("SSIM", ssim_v) + "'>" + f"{ssim_v:.4f}" + "</div>"
            "<div class='metric-note'>&#8805;0.85 = Excellent &nbsp;|&nbsp; 0.65&#8211;0.85 = Good &nbsp;|&nbsp; &lt;0.65 = Low</div>"
            "</div>"
            "<div class='metric-card'>"
            "<div class='metric-label'>Quality Rating</div>"
            "<div class='metric-value' style='font-size:1.1rem; padding-top:0.4rem;'>"
            "<span class='badge " + pb_cls + "'>" + pb_lbl + " PSNR</span><br><br>"
            "<span class='badge " + sb_cls + "'>" + sb_lbl + " SSIM</span>"
            "</div></div>"
            "</div>",
            unsafe_allow_html=True
        )
 
    # ── Visual comparison ─────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">&#128444; Visual Comparison</div>', unsafe_allow_html=True)
    if has_real:
        i1, i2, i3 = st.columns(3)
        with i1:
            st.image(sar_pil,  use_container_width=True)
            st.markdown('<p class="img-label">&#128225; Input SAR</p>', unsafe_allow_html=True)
        with i2:
            st.image(real_pil, use_container_width=True)
            st.markdown('<p class="img-label">&#127757; Real Optical</p>', unsafe_allow_html=True)
        with i3:
            st.image(pred_pil, use_container_width=True)
            st.markdown('<p class="img-label">&#129302; Predicted</p>', unsafe_allow_html=True)
    else:
        i1, i2 = st.columns(2)
        with i1:
            st.image(sar_pil,  use_container_width=True)
            st.markdown('<p class="img-label">&#128225; Input SAR</p>', unsafe_allow_html=True)
        with i2:
            st.image(pred_pil, use_container_width=True)
            st.markdown('<p class="img-label">&#129302; Predicted Optical</p>', unsafe_allow_html=True)
        st.info("&#128161; Upload the matching real optical image (s2) above to see PSNR / SSIM metrics and comparison table.")
    st.markdown('</div>', unsafe_allow_html=True)
 
    # ── Comparison table ──────────────────────────────────────────────────────
    if has_real:
        pred_arr = (np.clip(pred_rgb, 0, 1) * 255).astype(np.uint8)
        real_arr = (np.clip(real_rgb, 0, 1) * 255).astype(np.uint8)
 
        pb2_cls, pb2_lbl = badge_psnr(psnr_v)
        sb2_cls, sb2_lbl = badge_ssim(ssim_v)
 
        psnr_row = (
            "<tr><td><strong>PSNR</strong></td>"
            "<td colspan='4' style='color:#8b949e; font-style:italic'>full-image metric</td>"
            "<td><span class='badge " + pb2_cls + "'>" + f"{psnr_v:.2f}" + " dB &#8212; " + pb2_lbl + "</span></td></tr>"
        )
        ssim_row = (
            "<tr><td><strong>SSIM</strong></td>"
            "<td colspan='4' style='color:#8b949e; font-style:italic'>full-image metric</td>"
            "<td><span class='badge " + sb2_cls + "'>" + f"{ssim_v:.4f}" + " &#8212; " + sb2_lbl + "</span></td></tr>"
        )
 
        table_html = (
            "<div class='card'>"
            "<div class='card-title'>&#128202; Actual vs Predicted &#8212; Summary Table</div>"
            "<table class='compare-table'>"
            "<thead><tr>"
            "<th>Channel / Metric</th><th>Real (Actual)</th><th>Predicted</th>"
            "<th>Real Std Dev</th><th>Pred Std Dev</th><th>Difference</th>"
            "</tr></thead>"
            "<tbody>"
            + channel_row("Red",   real_arr, pred_arr, 0)
            + channel_row("Green", real_arr, pred_arr, 1)
            + channel_row("Blue",  real_arr, pred_arr, 2)
            + psnr_row
            + ssim_row
            + "</tbody></table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
 
    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    dl1, _, _ = st.columns([1, 1, 2])
    with dl1:
        st.download_button(
            label="&#128229; Download Predicted Image",
            data=to_bytes(pred_pil),
            file_name="predicted_optical.png",
            mime="image/png"
        )
 
else:
    st.markdown(
        "<div style='text-align:center; padding:4rem 2rem; color:#8b949e;'>"
        "<div style='font-size:3rem; margin-bottom:1rem;'>&#128752;</div>"
        "<div style='font-family:Space Mono,monospace; font-size:1rem;'>Upload a SAR image above to begin colorization</div>"
        "<div style='font-size:0.85rem; margin-top:0.5rem;'>Optionally upload the matching optical image to see PSNR / SSIM metrics</div>"
        "</div>",
        unsafe_allow_html=True
    )
