import torch
from PIL import Image
import torchvision.transforms as T
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
from model import EnsembleEncoder, Decoder   # from model.py

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_weights = "encoder.pth"
decoder_weights = "model_1.pth"   # you can also test with "decoder.pth"
sar_image_path = "test_sar.png"
output_path = "result.png"

# -----------------------------
# Load Models
# -----------------------------
encoder = EnsembleEncoder().to(device)
decoder = Decoder().to(device)

encoder.load_state_dict(torch.load(encoder_weights, map_location=device))
decoder.load_state_dict(torch.load(decoder_weights, map_location=device))

encoder.eval()
decoder.eval()

# -----------------------------
# Preprocess SAR image
# -----------------------------
img = Image.open(sar_image_path).convert("L")  # grayscale input
transform = T.Compose([
    T.Resize((224, 224)),  # use 224 because that’s what training used
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])
input_img = transform(img).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    f56, f28, f14, f7 = encoder(input_img)
    predicted_ab = decoder(f7, f14, f28, f56)

    # Reconstruct LAB image
    L_channel = (input_img[:, 0:1, :, :] + 1) * 0.5 * 100
    predicted_ab = ((predicted_ab + 1) * 0.5 * (127 + 128)) - 128
    predicted_lab = torch.cat([L_channel, predicted_ab], dim=1)
    predicted_lab = predicted_lab.squeeze().cpu().permute(1, 2, 0).numpy()

# -----------------------------
# Convert LAB → RGB
# -----------------------------
rgb_img = lab2rgb(predicted_lab)

# Save + Show
plt.imsave(output_path, rgb_img)
plt.imshow(rgb_img)
plt.axis("off")
plt.show()

print(f"✅ Saved optical output as {output_path}")
