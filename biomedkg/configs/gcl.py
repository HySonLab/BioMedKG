from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class GCLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    GCL_IN_DIMS : int
    GCL_HIDDEN_DIM : int
    GCL_OUT_DIM : int
    GCL_NUM_HIDDEN : int
    GCL_DROP_OUT : bool


gcl_settings = GCLSettings()