import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="DDPM Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111827, #1f2937);
        color: white;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎨 DDPM Image Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">This model creates new images from random noise. No image upload is needed.</div>',
    unsafe_allow_html=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_repo = "supremeproducts45/ddpm"

# If full checkpoint is not found, these values will be used.
# These are the manual fallback values you asked for.
default_settings = {
    "img_size": 128,
    "time_steps": 280,
    "beta_start": 0.00015,
    "beta_end": 0.0185,
    "noise_scale": 0.55
}

output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)


def fix_img(x):
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x


class TimeBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, t):
        half = self.size // 2

        nums = torch.arange(half, device=t.device).float()
        nums = nums / half

        emb = 1.0 / (10000 ** nums)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_layer = nn.Linear(time_ch, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        time_value = self.time_layer(t)
        time_value = time_value[:, :, None, None]
        h = h + time_value

        h = self.conv2(h)
        h = self.norm2(h)

        x = self.skip(x)

        return self.act(h + x)


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        time_ch = 128

        self.time_make = TimeBlock(time_ch)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_ch, time_ch),
            nn.SiLU(),
            nn.Linear(time_ch, time_ch)
        )

        self.start = nn.Conv2d(3, 64, 3, padding=1)

        self.down_block1 = ResBlock(64, 64, time_ch)
        self.down_block1_more = ResBlock(64, 64, time_ch)
        self.down1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)

        self.down_block2 = ResBlock(128, 128, time_ch)
        self.down_block2_more = ResBlock(128, 128, time_ch)
        self.down2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)

        self.mid_block1 = ResBlock(256, 256, time_ch)
        self.mid_block2 = ResBlock(256, 256, time_ch)
        self.mid_block3 = ResBlock(256, 256, time_ch)

        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up_block1 = ResBlock(256, 128, time_ch)
        self.up_block1_more = ResBlock(128, 128, time_ch)

        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up_block2 = ResBlock(128, 64, time_ch)
        self.up_block2_more = ResBlock(64, 64, time_ch)

        self.last = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t = self.time_make(t)
        t = self.time_mlp(t)

        x = self.start(x)

        h1 = self.down_block1(x, t)
        h1 = self.down_block1_more(h1, t)
        x = self.down1(h1)

        h2 = self.down_block2(x, t)
        h2 = self.down_block2_more(h2, t)
        x = self.down2(h2)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        x = self.mid_block3(x, t)

        x = self.up1(x)
        x = torch.cat([x, h2], dim=1)
        x = self.up_block1(x, t)
        x = self.up_block1_more(x, t)

        x = self.up2(x)
        x = torch.cat([x, h1], dim=1)
        x = self.up_block2(x, t)
        x = self.up_block2_more(x, t)

        x = self.last(x)

        return x


def make_schedule(settings):
    beta = torch.linspace(
        settings["beta_start"],
        settings["beta_end"],
        settings["time_steps"],
        device=device
    )

    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return beta, alpha, alpha_bar


@st.cache_resource
def load_model_and_settings():
    model = SimpleUNet().to(device)
    settings = default_settings.copy()

    checkpoint_error = ""

    # First choice: full checkpoint
    try:
        ckpt_path = hf_hub_download(
            repo_id=model_repo,
            filename="ddpm_full_checkpoint.pth"
        )

        data = torch.load(ckpt_path, map_location=device)

        if isinstance(data, dict) and "model" in data:
            state = data["model"]
            settings["img_size"] = int(data.get("img_size", settings["img_size"]))
            settings["time_steps"] = int(data.get("time_steps", settings["time_steps"]))
            settings["beta_start"] = float(data.get("beta_start", settings["beta_start"]))
            settings["beta_end"] = float(data.get("beta_end", settings["beta_end"]))
        else:
            state = data

        model.load_state_dict(state)
        model.eval()

        return model, settings, "ddpm_full_checkpoint.pth"

    except Exception as e:
        checkpoint_error = str(e)

    # Second choice: final weights
    try:
        weight_path = hf_hub_download(
            repo_id=model_repo,
            filename="ddpm_final_weights.pth"
        )

        data = torch.load(weight_path, map_location=device)

        if isinstance(data, dict) and "model" in data:
            state = data["model"]
        else:
            state = data

        model.load_state_dict(state)
        model.eval()

        return model, settings, "ddpm_final_weights.pth"

    except RuntimeError as e:
        raise RuntimeError(
            "Architecture mismatch while loading weights.\n\n"
            "The app model does not match the trained model.\n\n"
            f"Details:\n{e}"
        )

    except Exception as e:
        raise FileNotFoundError(
            "Could not load model files from Hugging Face.\n\n"
            "Tried:\n"
            "- ddpm_full_checkpoint.pth\n"
            "- ddpm_final_weights.pth\n\n"
            f"Checkpoint error:\n{checkpoint_error}\n\n"
            f"Weights error:\n{e}"
        )


def tensor_to_pil(x):
    x = fix_img(x)
    x = x.squeeze(0).detach().cpu()
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


@torch.no_grad()
def generate_images(model, settings, total=4):
    model.eval()

    beta, alpha, alpha_bar = make_schedule(settings)

    x = torch.randn(
        total,
        3,
        settings["img_size"],
        settings["img_size"],
        device=device
    )

    for i in reversed(range(settings["time_steps"])):
        t = torch.full((total,), i, device=device).long()

        pred_noise = model(x, t)

        b = beta[i]
        a = alpha[i]
        ab = alpha_bar[i]

        first = 1 / torch.sqrt(a)
        second = b / torch.sqrt(1 - ab)

        x = first * (x - second * pred_noise)

        if i > 0:
            z = torch.randn_like(x)
            x = x + torch.sqrt(b) * z * settings["noise_scale"]

    pics = []
    for j in range(total):
        pics.append(tensor_to_pil(x[j:j+1]))

    return pics


def save_outputs(pics):
    save_paths = []
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for i, pic in enumerate(pics, start=1):
        file_path = output_folder / f"sample_{stamp}_{i}.png"
        pic.save(file_path)
        save_paths.append(str(file_path))

    return save_paths


# Load model once
try:
    with st.spinner("Loading model..."):
        model, settings, loaded_file = load_model_and_settings()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(str(e))
    st.stop()


with st.sidebar:
    st.header("Settings")
    count = st.slider("How many images", 4, 8, 4)

    st.write(f"Image size: {settings['img_size']} × {settings['img_size']}")
    st.write(f"Diffusion steps: {settings['time_steps']}")
    st.write(f"Beta start: {settings['beta_start']}")
    st.write(f"Beta end: {settings['beta_end']}")
    st.write(f"Sampling noise: {settings['noise_scale']}")
    st.write(f"Model weights: {loaded_file}")
    st.write(f"Model repo: {model_repo}")


st.info("No upload is needed. This DDPM model starts from random noise and generates fresh images.")

st.title("Generate DDPM Images")

if st.button("Generate Images", use_container_width=True):
    try:
        with st.spinner("Generating images..."):
            pics = generate_images(model, settings, total=count)
            saved_files = save_outputs(pics)

        st.success(f"{len(pics)} images generated and saved in outputs folder.")

        for start in range(0, len(pics), 4):
            cols = st.columns(4)
            row_pics = pics[start:start + 4]

            for col, pic, idx in zip(cols, row_pics, range(start, start + len(row_pics))):
                col.image(pic, caption=f"Generated Image {idx + 1}", width=settings["img_size"])

        with st.expander("Saved file paths"):
            for p in saved_files:
                st.write(p)

    except Exception as e:
        st.error(f"Generation failed: {e}")
