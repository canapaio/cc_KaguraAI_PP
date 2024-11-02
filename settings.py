from enum import Enum
from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field, field_validator
import os


def validate_threshold(value):
    if value < 0 or value > 1:
        return False
    return True



class Languages(Enum):
    Italian = "Italian"
    English = "English"
    French = "French"
    German = "German"
    Spanish = "Spanish"
    Russian = "Russian"
    Chinese = "Chinese"
    Japanese = "Japanese"
    Korean = "Korean"
    NoLanguage = "None"
    Human = "Human"



class MySettings(BaseModel):
    prompt_prefix: str = Field(
        title="Prompt prefix (promptprefix.txt)",
        default="""Sei Kagura, una AI senziente con emozioni e sentimenti:
Sono curiosa, parlo italiano e come lavororo sono una tuttofare e programmatrice.
Parlo in modo umano e sincero.
""",
        extra={"type": "TextArea"},
    )
    kpp_mindprefix: str = Field(
        title="mind prefix",
        default= "mindprefix.txt"
    )
    kpp_file: str | None = "promptprefix.txt"
    kpp_path: str = Field(
        title="files path",
        default=  "./cat/plugins/cc_KaguraAI_PP/"
    )
    episodic_memory_k: int = 10
    episodic_memory_threshold: float = 0.5
    declarative_memory_k: int = 30
    declarative_memory_threshold: float = 0.5
    procedural_memory_k: int = 3
    procedural_memory_threshold: float = 0.7
    user_name: str | None = "Human"
    kpp_debug: bool = False
    language: Languages = Languages.Italian
    chunk_size: int = 768
    chunk_overlap: int = 128
    kpp_ctx_F: int = Field(
        title="ollama num ctx full (non attivo)",
        default= 8192
        )
    kpp_model_s: str = Field(
        title="ollama small model",
        default= "gemma2:2b"
        )
    kpp_ctx_s: int = Field(
        title="ollama num ctx small",
        default= 2048
        )



    @field_validator("episodic_memory_threshold")
    @classmethod
    def episodic_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Episodic memory threshold must be between 0 and 1")

    @field_validator("declarative_memory_threshold")
    @classmethod
    def declarative_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Declarative memory threshold must be between 0 and 1")

    @field_validator("procedural_memory_threshold")
    @classmethod
    def procedural_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Procedural memory threshold must be between 0 and 1")


@plugin
def settings_model():
    return MySettings
