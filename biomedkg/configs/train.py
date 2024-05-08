from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    VAL_RATIO : float
    TEST_RATIO : float
    BATCH_SIZE : int
    
    LEARNING_RATE : float
    EPOCHS : int
    SCHEDULER_TYPE : str
    WARM_UP_RATIO : float

    SEED : int
    OUT_DIR : str
    LOG_DIR : str
    VAL_EVERY_N_EPOCH : int

train_settings = TrainSettings()