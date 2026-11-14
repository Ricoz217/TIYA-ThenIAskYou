"""Forward category registry. Connection names and hostnames never select implementations."""

from typing import Callable

from .openai_chat import ChatCompletions
from .openai_responses import Responses
from .anthropic import Messages
from .openai_images import Images

REGISTRY: dict[str, Callable | type] = {
    "chat_completions": ChatCompletions,
    "responses_api": Responses,
    "anthropic_api": Messages,
    "image_generation": Images,
}


def register(category: str, factory: Callable | type):
    if not isinstance(category, str) or not callable(factory):
        raise TypeError("需要分类名和 Provider factory")

    REGISTRY[category] = factory


def get_provider(route):
    try:
        factory = REGISTRY[route.provider_type]

    except KeyError:
        raise ValueError(f"未注册的模型 Provider 分类: {route.provider_type}") from None

    return factory(route)
