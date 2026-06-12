"""
Task-02: Image Generation with Pre-trained Models
Uses Stable Diffusion (diffusers) locally, with a Gradio UI.
Falls back to a HuggingFace Inference API call if GPU is unavailable.
"""

import os
import io
import base64
import requests
import gradio as gr
from PIL import Image
import torch

# ── HuggingFace Inference API fallback ────────────────────────────────────
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")          # Optional
HF_MODEL_ID  = "stabilityai/stable-diffusion-2-1"
HF_API_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"


def generate_via_hf_api(prompt: str, negative_prompt: str = "") -> Image.Image:
    """Use HuggingFace Inference API (no GPU needed)."""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt or "blurry, bad quality, distorted",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise RuntimeError(
            f"HF API error {response.status_code}: {response.text[:300]}"
        )


def generate_via_diffusers(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> Image.Image:
    """Local Stable Diffusion generation using diffusers."""
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe.enable_attention_slicing()  # CPU memory optimization

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "blurry, bad quality, distorted, ugly",
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
    )
    return result.images[0]


def generate_image(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    use_api: bool,
) -> tuple:
    """Route generation to local diffusers or HF API."""
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."

    try:
        if use_api:
            img = generate_via_hf_api(prompt, negative_prompt)
            source = "HuggingFace Inference API"
        else:
            img = generate_via_diffusers(
                prompt, negative_prompt, steps, guidance_scale, width, height
            )
            source = "Local Stable Diffusion v1.5"

        return img, f"✅ Generated via {source}"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ── Prompt helpers ─────────────────────────────────────────────────────────
STYLE_TAGS = {
    "Photorealistic": "photorealistic, 8k, ultra detailed, sharp focus",
    "Oil Painting": "oil painting, thick brush strokes, classical art style",
    "Anime": "anime style, vibrant colors, cel shading, Studio Ghibli",
    "Watercolor": "watercolor painting, soft edges, pastel tones",
    "Cyberpunk": "cyberpunk, neon lights, rain, futuristic cityscape",
    "Sketch": "pencil sketch, black and white, fine lines, detailed drawing",
}


def enhance_prompt(base_prompt: str, style: str) -> str:
    tag = STYLE_TAGS.get(style, "")
    return f"{base_prompt}, {tag}" if tag else base_prompt


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="Task-02 · Image Generation",
        theme=gr.themes.Base(
            primary_hue="pink",
            secondary_hue="purple",
            font=gr.themes.GoogleFont("Syne"),
        ),
        css="""
        .gradio-container { max-width: 1100px; margin: auto; }
        #title { text-align: center; padding: 20px 0 10px; }
        """,
    ) as demo:

        gr.Markdown(
            """
# 🎨 Task-02 — Image Generation with Pre-trained Models
**Generate stunning images from text prompts using Stable Diffusion.**
            """,
            elem_id="title",
        )

        with gr.Tabs():
            with gr.TabItem("🖼️ Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_in = gr.Textbox(
                            label="Prompt",
                            placeholder="A futuristic cityscape at sunset…",
                            lines=3,
                            value="A serene mountain lake at golden hour, misty atmosphere",
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="blurry, distorted, low quality…",
                            lines=2,
                        )
                        style_dd = gr.Dropdown(
                            choices=list(STYLE_TAGS.keys()) + ["None"],
                            value="Photorealistic",
                            label="Art Style",
                        )
                        enhance_btn = gr.Button("✨ Enhance Prompt with Style", size="sm")
                        use_api_cb = gr.Checkbox(
                            label="Use HuggingFace API (no GPU needed — set HF_API_TOKEN env var)",
                            value=True,
                        )
                        steps_sl = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
                        guidance_sl = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                        with gr.Row():
                            width_sl = gr.Slider(256, 768, value=512, step=64, label="Width")
                            height_sl = gr.Slider(256, 768, value=512, step=64, label="Height")
                        gen_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        img_out = gr.Image(label="Generated Image", height=512)
                        status_out = gr.Textbox(label="Status", interactive=False)

                enhance_btn.click(
                    fn=enhance_prompt,
                    inputs=[prompt_in, style_dd],
                    outputs=prompt_in,
                )
                gen_btn.click(
                    fn=generate_image,
                    inputs=[prompt_in, neg_prompt, steps_sl, guidance_sl, width_sl, height_sl, use_api_cb],
                    outputs=[img_out, status_out],
                )

            with gr.TabItem("💡 Prompt Gallery"):
                gr.Markdown(
                    """
## Starter Prompts — Click to copy

| Category | Prompt |
|----------|--------|
| Nature | `A misty forest at dawn, rays of light through the trees, cinematic` |
| Portrait | `Portrait of a cyberpunk warrior, neon lights, rain, highly detailed` |
| Architecture | `Ancient Roman temple at sunset, golden hour, dramatic sky` |
| Fantasy | `Dragon soaring above snow-capped mountains, epic fantasy art` |
| Space | `Astronaut floating in deep space, nebula background, photorealistic` |
| Abstract | `Abstract fluid art, swirling colors, oil on canvas` |
                    """
                )

            with gr.TabItem("📖 About"):
                gr.Markdown(
                    """
## How It Works

| Step | Description |
|------|-------------|
| 1 | Text prompt is encoded by a **CLIP text encoder** |
| 2 | A random Gaussian noise image is created in **latent space** |
| 3 | A **U-Net denoiser** iteratively removes noise guided by the prompt |
| 4 | The **VAE decoder** converts latent vectors to pixel images |

### Key Parameters
- **Inference Steps**: More steps → higher quality, slower generation
- **Guidance Scale (CFG)**: Higher → image adheres more strictly to prompt
- **Negative Prompt**: Steers generation away from unwanted features

### Stack
- 🤗 `diffusers` · Stable Diffusion v1.5 · HF Inference API · Gradio
                    """
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
