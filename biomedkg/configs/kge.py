from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class KGESetting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    KGE_ENCODER_MODEL_NAME : str
    KGE_DECODER_MODEL_NAME : str
    KGE_IN_DIMS : int
    KGE_HIDDEN_DIM : int
    KGE_OUT_DIM : int
    KGE_NUM_HIDDEN : int
    KGE_NUM_HEAD : Optional[int] = 1
    KGE_DROP_OUT : bool


kge_settings = KGESetting()