from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/app"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_MODEL_NAME: str = "fraud-detector"
    MLFLOW_PRODUCTION_ALIAS: str = "production"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # Monitoring thresholds
    PSI_ALARM_THRESHOLD: float = 0.5
    PSI_MODERATE_THRESHOLD: float = 0.2
    PSI_MODERATE_COUNT: int = 3
    NULL_RATE_THRESHOLD: float = 0.2
    PREDICTION_DRIFT_THRESHOLD: float = 0.25
    MONITORING_WINDOW_ROWS: int = 1000
    MONITORING_INTERVAL_SECONDS: int = 30

    # Monitoring reference profile
    REFERENCE_PROFILE_PATH: str = "data/reference_profile.pkl"
    RAW_DATA_PATH: str = "data/raw/Base.csv"

    # Deployment gate
    AUC_IMPROVEMENT_MARGIN: float = 0.005
    PROMOTION_MARGIN: float = 0.01

    # Serving
    SERVE_PORT: int = 8000
    MODEL_REFRESH_INTERVAL_SECONDS: int = 60


settings = Settings()
