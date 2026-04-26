import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from PIL import Image
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
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎨 DDPM Image Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">This DDPM model generates new images from random noise. No image upload is needed.</div>',
    unsafe_allow_html=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 64
time_steps = 300
beta_start = 1e-4
beta_end = 0.02

beta = torch.linspace(beta_start, beta_end, time_steps).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


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


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="supremeproducts45/ddpm",
        filename="ddpm_final_weights.pth"
    )

    model = SimpleUNet().to(device)

    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state)
    model.eval()
    return model


def tensor_to_pil(x):
    x = fix_img(x)
    x = x.squeeze(0).detach().cpu()
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


@torch.no_grad()
def generate_images(model, total=1):
    model.eval()
    x = torch.randn(total, 3, img_size, img_size).to(device)

    for i in reversed(range(time_steps)):
        t = torch.full((x.shape[0],), i, device=device).long()

        pred_noise = model(x, t)

        b = beta[i]
        a = alpha[i]
        ab = alpha_bar[i]

        if i > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - ab)) * pred_noise) + torch.sqrt(b) * z

    pics = []
    for j in range(total):
        pics.append(tensor_to_pil(x[j:j+1]))

    return pics


with st.sidebar:
    st.header("Settings")
    count = st.slider("How many images", 1, 4, 1)
    st.write(f"Image size: {img_size} × {img_size}")
    st.write(f"Diffusion steps: {time_steps}")
    st.write("Model repo: supremeproducts45/ddpm")

st.info("This model does not need image upload. Just click the button below to generate new images from random noise.")

st.title("Generate DDPM Images")

if st.button("Generate Images", use_container_width=True):
    try:
        with st.spinner("Loading model and generating images..."):
            model = load_model()
            results = generate_images(model, total=count)

        st.success("Done!")

        cols = st.columns(count)
        for i in range(count):
            cols[i].image(results[i], caption=f"Generated Image {i+1}", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
