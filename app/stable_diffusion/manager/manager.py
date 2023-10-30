import os
import typing as T
import torch

torch.backends.cudnn.benchmark = True
import sys
from random import randint
from service_streamer import ThreadedStreamer
from diffusers import DiffusionPipeline

from app.stable_diffusion.manager.schema import (
    InpaintTask,
    Text2ImageTask,
    Image2ImageTask,
)
from core.settings import get_settings

from core.utils.convert_script import conver_ckpt_to_diff

from functools import lru_cache

env = get_settings()

_StableDiffusionTask = T.Union[
    Text2ImageTask,
    Image2ImageTask,
    InpaintTask,
]


@lru_cache()
def build_pipeline(repo: str, device: str, enable_attention_slicing: bool):

    # convert ckpt to diffusers
    if repo.lower().endswith(".ckpt") and os.path.exists(repo):
        dump_path = repo[:-5]
        repo = conver_ckpt_to_diff(ckpt_path=repo, dump_path=dump_path)

    # stable_diffusion_txt2img = AutoPipelineForText2Image.from_pretrained(
    #     repo,
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     use_safetensors=True,
    #     # revision="fp16",
    #     # custom_pipeline="lpw_stable_diffusion",
    # )
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    # stable_diffusion_img2img = AutoPipelineForImage2Image.from_pipe(
    #     stable_diffusion_txt2img
    # )
    #
    # stable_diffusion_txt2img.scheduler = DPMSolverMultistepScheduler.from_config(stable_diffusion_txt2img.scheduler.config)
    # stable_diffusion_txt2img.safety_checker = lambda images, clip_input: (images, False)
    #
    # stable_diffusion_inpaint = AutoPipelineForInpainting.from_pipe(stable_diffusion_txt2img)
    #
    # if enable_attention_slicing:
    #     stable_diffusion_txt2img.enable_attention_slicing()

    # pipe.enable_model_cpu_offload()
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # stable_diffusion_txt2img = stable_diffusion_txt2img.to(device)
    # stable_diffusion_txt2img.load_lora_weights("./lora/anime_sdxl_v1.safetensors", weight_name="anime_sdxl_v1.safetensors",
    #                        adapter_name='anime')
    # active_adapters = stable_diffusion_txt2img.get_active_adapters()

    # stable_diffusion_img2img = stable_diffusion_img2img.to(device)
    # stable_diffusion_inpaint = stable_diffusion_inpaint.to(device)

    # logger.info(f"{active_adapters}")
    return dict(
        text2img=base,
        img2img=refiner,
        inpaint=refiner,
    )


build_pipeline(
    repo=env.MODEL_ID,
    device=env.CUDA_DEVICE,
    enable_attention_slicing=env.ENABLE_ATTENTION_SLICING,
)


class StableDiffusionManager:
    def __init__(self):
        self.pipe = build_pipeline(
            repo=env.MODEL_ID,
            device=env.CUDA_DEVICE,
            enable_attention_slicing=env.ENABLE_ATTENTION_SLICING,
        )

    @torch.inference_mode()
    def predict(
        self,
        batch: T.List[_StableDiffusionTask],
    ):
        task = batch[0]
        pipeline = self.pipe
        if isinstance(task, Text2ImageTask):
            pipeline = self.pipe['text2img']
        elif isinstance(task, Image2ImageTask):
            pipeline = self.pipe['img2img']
        elif isinstance(task, InpaintTask):
            pipeline = self.pipe['inpaint']
        else:
            raise NotImplementedError

        device = env.CUDA_DEVICE

        n_steps = 40
        high_noise_frac = 0.8
        task = task.dict()
        image = pipeline(
            prompt=task['prompt'],
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        images = self.pipe['img2img'](
            prompt=task['prompt'],
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images

        # generator = self._get_generator(task, device)
        # task = task.dict()
        # del task["seed"]
        # images = pipeline(**task, generator=generator).images
        # if device != "cpu":
        #     torch.cuda.empty_cache()

        return [images]

    def _get_generator(self, task: _StableDiffusionTask, device: str):
        generator = torch.Generator(device=device)
        seed = task.seed
        seed = seed if seed else randint(1, sys.maxsize)
        seed = seed if seed > 0 else randint(1, sys.maxsize)
        generator.manual_seed(seed)
        return generator


@lru_cache(maxsize=1)
def build_streamer() -> ThreadedStreamer:
    manager = StableDiffusionManager()
    streamer = ThreadedStreamer(
        manager.predict,
        batch_size=1,
        max_latency=0,
    )
    return streamer
