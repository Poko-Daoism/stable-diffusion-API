import os
import typing as T

from pydantic import BaseSettings


class ModelSetting(BaseSettings):
    MODEL_ID: str = "CompVis/stable-diffusion-v1-4"


class DeviceSettings(BaseSettings):
    CUDA_DEVICE = "cuda"
    CUDA_DEVICES = [0]


class MicroBatchSettings(BaseSettings):
    MB_BATCH_SIZE = 2


class Settings(
    ModelSetting,
    DeviceSettings,
    MicroBatchSettings,
):
    HUGGINGFACE_TOKEN: str
    IMAGESERVER_URL: str
    SAVE_DIR: str = "static"

    CORS_ALLOW_ORIGINS: T.List[str] = ["*"]
