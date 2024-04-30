from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class KGESetting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    ENCODER_MODEL_NAME : str
    DECODER_MODEL_NAME : str
    IN_DIMS : int
    HIDDEN_DIM : int
    OUT_DIM : int
    NUM_HIDDEN : int
    NUM_HEAD : Optional[int] = 1
    DROP_OUT : bool


kge_settings = KGESetting()