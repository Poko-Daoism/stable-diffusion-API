import os
import typing as T
import torch
from loguru import logger

torch.backends.cudnn.benchmark = True
import sys
from random import randint
from service_streamer import ThreadedStreamer
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

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

    stable_diffusion_txt2img = DiffusionPipeline.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        # revision="fp16",
        # custom_pipeline="lpw_stable_diffusion",
    )

    stable_diffusion_txt2img.scheduler = DPMSolverMultistepScheduler.from_config(stable_diffusion_txt2img.scheduler.config)
    # pipe.safety_checker = lambda images, clip_input: (images, False)

    if enable_attention_slicing:
        stable_diffusion_txt2img.enable_attention_slicing()

    # pipe.enable_model_cpu_offload()
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    stable_diffusion_txt2img = stable_diffusion_txt2img.to(device)
    stable_diffusion_txt2img.load_lora_weights("./lora/anime_sdxl_v1.safetensors", weight_name="anime_sdxl_v1.safetensors",
                           adapter_name='anime')
    active_adapters = stable_diffusion_txt2img.get_active_adapters()

    # img to img
    components = stable_diffusion_txt2img.components
    stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(**components)
    stable_diffusion_inpaint = StableDiffusionInpaintPipeline(**components)

    logger.info(f"{active_adapters}")
    return dict(
        text2img=stable_diffusion_txt2img,
        img2img=stable_diffusion_img2img,
        inpaint=stable_diffusion_inpaint,
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

        generator = self._get_generator(task, device)
        task = task.dict()
        del task["seed"]
        images = pipeline(**task, generator=generator).images
        if device != "cpu":
            torch.cuda.empty_cache()

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
