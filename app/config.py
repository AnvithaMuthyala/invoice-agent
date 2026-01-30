import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    EVALUATION_MODEL: str = os.getenv("EVALUATION_MODEL", "gemini-1.5-flash")


config = Config()
