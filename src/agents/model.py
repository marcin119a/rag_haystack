from agno.models.openai import OpenAIChat

from settings import settings


def build_model() -> OpenAIChat:
    return OpenAIChat(id=settings.openai_model, api_key=settings.openai_api_key)
