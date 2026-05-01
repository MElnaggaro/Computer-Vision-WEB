from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Computer Vision API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Vision Settings
    FACE_RECOGNITION_MODEL: str = "default_model"
    CONFIDENCE_THRESHOLD: float = 0.6
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
