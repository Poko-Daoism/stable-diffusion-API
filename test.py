from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

with torch.autocast("cuda"):
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipe(prompt=prompt).images[0]
    image[0].save("astronaut1.png")
