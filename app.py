import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import io

# Title
st.title("Image Generator")
st.write("Enter a prompt and generate an AI image.")

# Load Model (only once to optimize performance)
@st.cache_resource()
def load_model():
    global pipe
    if 'pipe' not in st.session_state:
        st.session_state.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        st.session_state.pipe.to("cuda")
    return st.session_state.pipe

pipe = load_model()

# User Input
prompt = st.text_area("Enter your prompt:","A majestic dragon with glowing emerald scales soars over a misty castle")

generate_btn = st.button("Generate Image")

if generate_btn:
    with st.spinner("Generating Image... Please wait!"):
        images = pipe(prompt=prompt).images[0]
        img_bytes = io.BytesIO()
        images.save(img_bytes, format="JPEG")
        st.image(images, caption="Generated Image", use_column_width=True)
        st.download_button("Download Image", img_bytes.getvalue(), "generated_image.jpg", "image/jpeg")
