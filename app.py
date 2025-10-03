import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
from skimage.color import lab2rgb
import numpy as np
import io

# Import your model
from model import EnsembleEncoder, Decoder

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_weights = "encoder.pth"
decoder_weights = "decoder.pth"

# Load Models
encoder = EnsembleEncoder().to(device)
decoder = Decoder().to(device)

encoder.load_state_dict(torch.load(encoder_weights, map_location=device))
decoder.load_state_dict(torch.load(decoder_weights, map_location=device))

encoder.eval()
decoder.eval()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌌 SAR → Optical Image Translator")

uploaded_file = st.file_uploader("Upload a SAR image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess image
    img = Image.open(uploaded_file).convert("L")  # SAR is grayscale
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    input_img = transform(img).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

    # Run inference
    with torch.no_grad():
        f56, f28, f14, f7 = encoder(input_img)
        predicted_ab = decoder(f7, f14, f28, f56)

        L_channel = (input_img[:, 0:1, :, :] + 1) * 0.5 * 100
        predicted_ab = ((predicted_ab + 1) * 0.5 * (127 + 128)) - 128
        predicted_lab = torch.cat([L_channel, predicted_ab], dim=1)
        predicted_lab = predicted_lab.squeeze().cpu().permute(1, 2, 0).numpy()

    rgb_img = lab2rgb(predicted_lab)

    # Show original + result
    st.subheader("Input SAR Image")
    st.image(img, use_column_width=True)

    st.subheader("Generated Optical Image")
    st.image(rgb_img, use_column_width=True)

    # Allow download
    result_pil = Image.fromarray((rgb_img * 255).astype(np.uint8))
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Optical Image",
        data=byte_im,
        file_name="optical_result.png",
        mime="image/png"
    )
