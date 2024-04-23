from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainKGESetting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env-kge-train", extra="allow")

    NUM_VAL : float
    NUM_TEST : float
    BATCH_SIZE : int
    
    STEP_PER_EPOCH : int
    LEARNING_RATE : float
    EPOCHS : int

kge_train_settings = TrainKGESetting()