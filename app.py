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

# -------------------------------------------------------
# Page config
# -------------------------------------------------------

st.set_page_config(
    page_title="ColorSAR",
    page_icon="🛰️",
    layout="wide"
)

# -------------------------------------------------------
# Load models
# -------------------------------------------------------

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EnsembleEncoder().to(device)
    decoder = Decoder().to(device)

    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder, device


encoder, decoder, device = load_models()

# -------------------------------------------------------
# Preprocessing
# -------------------------------------------------------

def preprocess_sar(img_pil):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    img = transform(img_pil).unsqueeze(0)
    img = img.repeat(1,3,1,1)

    return img.to(device)

# -------------------------------------------------------
# Inference
# -------------------------------------------------------

def run_inference(tensor):

    with torch.no_grad():

        f56, f28, f14, f7 = encoder(tensor)

        ab = decoder(f7, f14, f28, f56)

        L = (tensor[:,0:1] + 1) * 0.5 * 100

        ab = ((ab + 1) * 0.5 * 255) - 128

        lab = torch.cat([L,ab],dim=1)

        lab = lab.squeeze().cpu().permute(1,2,0).numpy()

        rgb = lab2rgb(lab)

    return rgb

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

def compute_metrics(real_rgb, pred_rgb):

    real = (np.clip(real_rgb,0,1)*255).astype(np.uint8)
    pred = (np.clip(pred_rgb,0,1)*255).astype(np.uint8)

    psnr_val = psnr(real,pred,data_range=255)

    ssim_val = ssim(real,pred,channel_axis=2,data_range=255)

    return psnr_val, ssim_val

# -------------------------------------------------------
# UI
# -------------------------------------------------------

st.title("🛰️ SAR Image Colorization")
st.write("Upload a SAR image to generate its optical colorized version")

col1,col2 = st.columns(2)

with col1:
    sar_file = st.file_uploader(
        "Upload SAR image",
        type=["png","jpg","jpeg"]
    )

with col2:
    opt_file = st.file_uploader(
        "Upload real optical image (optional)",
        type=["png","jpg","jpeg"]
    )

# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------

if sar_file:

    sar_pil = Image.open(sar_file).convert("L")

    tensor = preprocess_sar(sar_pil)

    pred_rgb = run_inference(tensor)

    pred_img = Image.fromarray(
        (np.clip(pred_rgb,0,1)*255).astype(np.uint8)
    )

    st.subheader("Prediction")

    colA,colB = st.columns(2)

    with colA:
        st.image(sar_pil, caption="Input SAR")

    with colB:
        st.image(pred_img, caption="Predicted Optical")

    # -------------------------------------------------------
    # Metrics if reference available
    # -------------------------------------------------------

    if opt_file:

        real_img = Image.open(opt_file).convert("RGB").resize((224,224))

        real_rgb = np.array(real_img).astype(np.float32)/255.0

        psnr_val, ssim_val = compute_metrics(real_rgb,pred_rgb)

        st.subheader("Evaluation Metrics")

        m1,m2 = st.columns(2)

        with m1:
            st.metric("PSNR", f"{psnr_val:.2f} dB")

        with m2:
            st.metric("SSIM", f"{ssim_val:.4f}")

        st.image(real_img, caption="Real Optical")

# -------------------------------------------------------
# Download
# -------------------------------------------------------

    buf = io.BytesIO()
    pred_img.save(buf, format="PNG")

    st.download_button(
        "Download Predicted Image",
        data=buf.getvalue(),
        file_name="predicted_optical.png",
        mime="image/png"
    )

else:

    st.info("Upload a SAR image to start colorization.")
