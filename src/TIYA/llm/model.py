from __future__ import annotations
import asyncio
import copy
import datetime
import inspect
import time
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Awaitable
from uuid import uuid4
from TIYA.time_id import next_time_id
from .config_model import (
    ChatConfig as ChatConfig,
    LLMConfig as LLMConfig,
    FilesConfig as FilesConfig,
    RequestPolicy as RequestPolicy,
)


class ContextOverflowError(Exception):
    """超出模型context窗口错误"""

    pass


class ContextEmptyError(Exception):
    pass


class PromptType(Enum):
    BASE = "BASE"
    SYSTEM = "SYSTEM"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    TOOL_IN = "TOOL_IN"
    TOOL_CALL = "TOOLCALL"
    TOOL_RESP = "TOOL_RESP"
    TOKEN = "TOKEN"


class BasePrompt:
    def __init__(self, role: str, contain_type: str, create_time: str = None):
        if create_time is None:
            create_time = str(time.time())

        self.role: str = role
        self.contain_type: str = contain_type
        self.time: str = create_time
        self.id: str = str(next_time_id())
        self.native: dict = {}

    def to_dict(self) -> dict:
        return {
            "type": PromptType.BASE.value,
            "role": self.role,
            "contain_type": self.contain_type,
            "time": self.time,
            "id": self.id,
            **({"native": copy.deepcopy(self.native)} if self.native else {}),
        }

    def __eq__(self, other):
        if isinstance(other, BasePrompt):
            return self.role == other.role and self.contain_type == other.contain_type

        else:
            return False

    def __str__(self):
        dt = datetime.datetime.fromtimestamp(float(self.time))
        output = f"[{self.role}] {dt:%Y-%m-%d %H:%M:%S}: "
        return output

    def __add__(self, other):
        return Prompts(self, other)


class SystemPrompt(BasePrompt):
    def __init__(self, text: str, create_time: str = None):
        super().__init__("system", "TEXT", create_time)
        self.text = text

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.SYSTEM.value
        base_dict["data"] = {"text": self.text}
        return base_dict

    def copy(self):
        return self.__copy__()

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = SystemPrompt(data["data"]["text"], data["time"])
        new_prompt.id = data["id"]
        return new_prompt

    def __copy__(self):
        return SystemPrompt(self.text, create_time=self.time)

    def __eq__(self, other):
        if isinstance(other, SystemPrompt):
            return super().__eq__(other) and self.text == other.text

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += self.text
        output += "\n"
        return output

    def __bool__(self):
        return bool(self.text)


class TextPrompt(BasePrompt):
    def __init__(
        self, role: str, text: str, reasoning_content: str = "", create_time: str = None
    ):
        super().__init__(role, "TEXT", create_time)
        self.text: str = text
        self.reasoning_content: str = reasoning_content

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TEXT.value
        base_dict["data"] = {
            "text": self.text,
            "reasoning_content": self.reasoning_content,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = TextPrompt(
            role=data["role"],
            text=data["data"].get("text", ""),
            reasoning_content=data["data"].get("reasoning_content", ""),
            create_time=data["time"],
        )
        new_prompt.id = data["id"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, TextPrompt):
            return (
                super().__eq__(other)
                and self.text == other.text
                and self.reasoning_content == other.reasoning_content
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += self.text
        output += "\n"
        return output

    def __bool__(self):
        return bool(self.text)


class ImagePrompt(BasePrompt):
    def __init__(
        self,
        role: str,
        image: str | Path | bytes,
        name: str = "",
        create_time: str = None,
    ):
        super().__init__(role, "IMAGE", create_time)
        self.name: str = name  # 图片的唯一ID，用于查询缓存
        self.image: list[str] = []  # base64字符串哈希列表，节省内存
        self.image_input = image
        self.media_descriptor: str = ""
        self.prepared_input: str = ""
        self.resource: MediaRef | None = None

    def to_dict(self) -> dict:
        if not isinstance(self.image_input, Path):
            from TIYA.file_cache import get_file_path, add_file

            origin_image_path = get_file_path(
                add_file(self.image_input)
            )  # 这是保底，正常不会触发

        else:
            origin_image_path = self.image_input

        base_dict = super().to_dict()
        base_dict["type"] = PromptType.IMAGE.value
        base_dict["data"] = {
            "name": self.name,
            "image": self.image,
            "image_input": str(origin_image_path),
            "media_descriptor": self.media_descriptor,
            "prepared_input": self.prepared_input,
            "resource": asdict(self.resource) if self.resource else None,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ImagePrompt(
            data["role"],
            Path(data["data"]["image_input"]),
            data["data"]["name"],
            data["time"],
        )
        new_prompt.id = data["id"]
        new_prompt.image = data["data"]["image"]
        new_prompt.media_descriptor = data["data"].get("media_descriptor", "")
        new_prompt.prepared_input = data["data"].get("prepared_input", "")
        if data["data"].get("resource"):
            new_prompt.resource = MediaRef(**data["data"]["resource"])
        return new_prompt

    async def load_image(self, cache: dict[str, str]) -> bool:
        from .legacy_media import (
            _IMAGE_NAME_MAPPING,
            _load_image_hashes,
            handle_unload_image,
        )
        from TIYA.file_cache import add_file_async, get_file_path_async

        if self.name:
            hash_name_list = _IMAGE_NAME_MAPPING.get(self.name, [])
            if hash_name_list and await _load_image_hashes(hash_name_list, cache):
                self.image = list(hash_name_list)
                return True

            _IMAGE_NAME_MAPPING.pop(self.name, None)

        self.image = []
        try:
            hash_name = await add_file_async(self.image_input)  # 入库
            hash_name_list = await handle_unload_image(self.image_input, cache)
            self.image_input = await get_file_path_async(hash_name)

        except (FileNotFoundError, OSError, ValueError):
            return False

        if not hash_name_list:
            return False

        self.image = hash_name_list
        if self.name:
            _IMAGE_NAME_MAPPING[self.name] = list(self.image)

        return True

    def __eq__(self, other):
        if isinstance(other, ImagePrompt):
            return (
                bool(self.image)
                and super().__eq__(other)
                and self.name == other.name
                and self.image == other.image
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[图片;hash_id: {self.image}, name: {self.name}]"
        output += "\n"
        return output


class ToolInput(BasePrompt):
    def __deepcopy__(self, memo):
        # Bound methods belong to business objects, often containing locks/clients.
        result = object.__new__(type(self))
        memo[id(self)] = result
        result.__dict__ = {
            k: v if k == "function" else copy.deepcopy(v, memo)
            for k, v in vars(self).items()
        }
        return result

    def __init__(
        self,
        function: Callable | Awaitable,
        function_name: str = None,
        create_time: str = None,
    ):
        if function_name is None:
            function_name = function.__name__

        super().__init__("function", "FUNCTION", create_time)
        self.function: Callable | Awaitable = function
        self.function_name: str = function_name
        if inspect.iscoroutinefunction(function):
            self.function_type = "ASYNC"

        else:
            self.function_type = "SYNC"

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TOOL_IN.value
        base_dict["data"] = {
            "function_name": self.function_name,
            "function_type": self.function_type,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ToolInput(
            data["data"]["function"], data["data"]["function_name"], data["time"]
        )
        new_prompt.id = data["id"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, ToolInput):
            return (
                super().__eq__(other)
                and self.function is other.function
                and self.function_type == other.function_type
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[TOOL输入; function_name: {self.function_name}]"
        output += "\n"
        return output


class ToolCall(BasePrompt):
    def __init__(self, function_name: str, call_id: str, arguments: Any):
        super().__init__("assistant", "FUNCTION")
        self.function_name: str = function_name
        self.call_id: str = call_id
        self.arguments: Any = arguments

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TOOL_CALL.value
        base_dict["data"] = {
            "function_name": self.function_name,
            "call_id": self.call_id,
            "arguments": self.arguments,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ToolCall(
            data["data"]["function_name"],
            data["data"]["call_id"],
            data["data"]["arguments"],
        )
        new_prompt.id = data["id"]
        new_prompt.time = data["time"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, ToolCall):
            return (
                super().__eq__(other)
                and self.function_name == other.function_name
                and self.call_id == other.call_id
                and self.arguments == other.arguments
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[TOOL CALL; function_name: {self.function_name}, arguments: {self.arguments}]"
        output += "\n"
        return output


class ToolResponse(BasePrompt):
    def __init__(
        self, function_name: str, call_id: str, response: Any, create_time: str = None
    ):
        super().__init__("tool", "FUNCTION", create_time)
        self.function_name = function_name
        self.call_id: str = call_id
        self.response = response

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TOOL_RESP.value
        base_dict["data"] = {
            "function_name": self.function_name,
            "call_id": self.call_id,
            "response": self.response,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ToolResponse(
            data["data"]["function_name"],
            data["data"]["call_id"],
            data["data"]["response"],
            data["time"],
        )
        new_prompt.id = data["id"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, ToolResponse):
            return (
                super().__eq__(other)
                and self.function_name == other.function_name
                and self.call_id == other.call_id
                and self.response == other.response
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[TOOL RESPONSE; function_name: {self.function_name}, response: {self.response}]"
        output += "\n"
        return output


@dataclass(slots=True)
class TokenUsage:
    """统一 token 统计结构。"""

    input_t: int = 0
    output_t: int = 0
    cache_hit_t: int = 0
    total_t: int = 0
    model: str = ""
    id: str = field(default_factory=lambda: str(next_time_id()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_t": self.input_t,
            "output_t": self.output_t,
            "cache_hit_t": self.cache_hit_t,
            "total_t": self.total_t,
            "model": self.model,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict):
        new_usage = TokenUsage(
            data.get("input_t", 0),
            data.get("output_t", 0),
            data.get("cache_hit_t", 0),
            data.get("total_t", 0),
            data.get("model", ""),
            data.get("id", str(next_time_id())),
        )
        return new_usage

    def clear(self):
        self.input_t = 0
        self.output_t = 0
        self.cache_hit_t = 0
        self.total_t = 0
        self.model = ""

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return TokenUsage(
            self.input_t, self.output_t, self.cache_hit_t, self.total_t, self.model
        )

    def __add__(self, other) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            raise TypeError(f"'+' doesn't support type: {type(other)}")

        if self.model != other.model:
            if self.model:
                raise ValueError("Not Same Model")

            else:
                self.model = other.model

        new_input_t = self.input_t + other.input_t
        new_output_t = self.output_t + other.output_t
        new_cache_hit_t = self.cache_hit_t + other.cache_hit_t
        new_total_t = self.total_t + other.total_t
        return TokenUsage(
            new_input_t, new_output_t, new_cache_hit_t, new_total_t, self.model
        )

    def __sub__(self, other) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            raise TypeError(f"'-' doesn't support type: {type(other)}")

        if self.model != other.model:
            if self.model:
                raise ValueError("Not Same Model")

        new_input_t = max(0, self.input_t - other.input_t)
        new_output_t = max(0, self.output_t - other.output_t)
        new_cache_hit_t = max(0, self.cache_hit_t - other.cache_hit_t)
        new_total_t = max(0, self.total_t - other.total_t)
        return TokenUsage(
            new_input_t, new_output_t, new_cache_hit_t, new_total_t, self.model
        )

    def __gt__(self, other):
        if not isinstance(other, TokenUsage):
            raise TypeError(f"'>' doesn't support type: {type(other)}")

        if self.model != other.model:
            if self.model:
                raise ValueError("Not Same Model")

        return self.total_t > other.total_t

    def __lt__(self, other):
        if not isinstance(other, TokenUsage):
            raise TypeError(f"'>' doesn't support type: {type(other)}")

        if self.model != other.model:
            if self.model:
                raise ValueError("Not Same Model")

        return self.total_t < other.total_t

    def __bool__(self):
        return bool(self.total_t)


class Prompts:
    def __init__(
        self, *args, role: str = "", create_time: str = None, duration: float = 0.0
    ):
        if create_time is None:
            create_time = time.time()

        self.prompts: list[BasePrompt] = []
        self.role = role
        self.time = create_time  # 最新一条消息的时间
        self._duration: float = duration  # 耗时
        self.usage: TokenUsage | None = None
        if args:
            self._initiate_prompts(args)

    def _add_usage(self, usage: TokenUsage):
        if self.usage is None:
            self.usage = usage

        else:
            self.usage += usage

    def _unpack_prompts(self, prompts, roles: set):
        if isinstance(prompts, Prompts):
            for p in prompts.prompts:
                roles.add(p.role)
                self.prompts.append(p)

            if prompts.usage is not None:
                self._add_usage(prompts.usage)

            self.time = prompts.time

        elif isinstance(prompts, (list, tuple)):
            for p in prompts:
                self._unpack_prompts(p, roles)

        elif isinstance(prompts, BasePrompt):
            roles.add(prompts.role)
            self.prompts.append(prompts)
            self.time = prompts.time

        elif isinstance(prompts, str):  # 仅提供对文本内容的快捷转换
            self._text2prompt(prompts, roles)

        elif isinstance(prompts, TokenUsage):
            self._add_usage(prompts)

        else:
            raise TypeError(f"不支持{type(prompts)}生成Prompts")

    def _initiate_prompts(self, *args):
        roles = set()
        for prompts in args:
            self._unpack_prompts(prompts, roles)

        if len(roles) == 1:
            self.role = next(iter(roles))

        elif len(roles) > 1:
            self.role = "MIX"

    def _text2prompt(self, text: str, roles: set):
        if len(roles) == 1:
            role = next(iter(roles))

        elif self.role:
            role = self.role

        else:
            raise ValueError("需要指定一个role")

        self.prompts.append(TextPrompt(role, text))

    def to_dict(self) -> dict:
        prompts = [prompt.to_dict() for prompt in self.prompts]
        usage = self.usage.to_dict() if self.usage is not None else {}
        return {
            "prompts": prompts,
            "role": self.role,
            "time": self.time,
            "duration": self._duration,
            "usage": usage,
        }

    @property
    def duration(self) -> float:
        if self._duration:
            return self._duration

        else:
            if len(self.prompts) > 1:
                return max(
                    float(self.prompts[-1].time) - float(self.prompts[0].time), 0.0
                )

            else:
                return 0.0

    def append(self, addition: BasePrompt | Prompts | TokenUsage | str | list) -> None:
        roles = set()
        self._unpack_prompts(addition, roles)
        if len(roles) == 1:
            new_role = next(iter(roles))
            if self.role:
                if self.role != new_role:
                    self.role = "MIX"

            else:
                self.role = new_role

        else:
            self.role = "MIX"

    def extend(self, addition: BasePrompt | Prompts | str | list) -> None:
        return self.append(addition)

    def discard_prompt_ids(self, prompt_ids: set[str]) -> int:
        if not prompt_ids:
            return 0

        original_count = len(self.prompts)
        self.prompts = [
            prompt for prompt in self.prompts if prompt.id not in prompt_ids
        ]
        return original_count - len(self.prompts)

    def copy(self):
        return self.__copy__()

    def __str__(self):
        output = ""
        for prompt in self.prompts:
            output += str(prompt)

        return output

    def __contains__(self, item):
        if isinstance(item, BasePrompt):
            for p in self.prompts:
                if p == item:
                    return True

            return False

        else:
            return False

    def __iter__(self):
        return iter(self.prompts)

    def __reversed__(self):
        return reversed(self.prompts)

    def __add__(self, other):
        if isinstance(other, (Prompts, BasePrompt, str)):
            return Prompts(self, other)

        else:
            raise TypeError(f"Prompts add doesn't support {type(other)}")

    def __sub__(self, other):
        origin = self.prompts.copy()
        if isinstance(other, Prompts):
            sub = other.prompts.copy()
            for p in origin[:]:
                if p in sub:
                    origin.remove(p)

            if self.usage and other.usage:
                new_usage = self.usage - other.usage
                return Prompts(origin, new_usage)

            return Prompts(origin)

        elif isinstance(other, BasePrompt):
            for p in origin[:]:
                if p == other:
                    origin.remove(p)
                    return Prompts(origin)

            return Prompts(origin)

        elif isinstance(other, TokenUsage):
            if self.usage is not None:
                new_prompts = self.copy()
                new_usage = self.usage - other
                new_prompts.usage = new_usage
                return new_prompts

            raise ValueError("host Prompts doesn't has TokenUsage yet")

        else:
            raise TypeError(f"Prompts sub doesn't support {type(other)}")

    def __bool__(self):
        return bool(self.prompts)

    def __copy__(self):
        return copy.deepcopy(self)

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise KeyError("index must be int")

        return self.prompts[item]


class Context:
    def __init__(self, *args):
        self.session_id = uuid4().hex
        self._system_prompt: SystemPrompt | None = None
        self._messages: list[BasePrompt] = []
        self._rounds: dict[str, Prompts] = {}
        self._tools: list[ToolInput] = []
        self._usage_accumulation: TokenUsage = TokenUsage()
        if args:
            self._initiate(args)

    def _initiate(self, args: tuple):
        new_round: dict[str, list] = {"prompts": [], "usage": []}
        self._unpack_prompts(args, new_round)
        if new_round:
            self._submit_round(new_round)

    def _add_usage(self, usage: TokenUsage):
        if not self._usage_accumulation.model:
            self._usage_accumulation.model = usage.model

        self._usage_accumulation += usage

    def _submit_round(self, round_list: dict):
        save_round = Prompts(round_list["prompts"], round_list["usage"])
        save_round.role = "ROUND"
        self._rounds[save_round.time] = save_round
        round_list.clear()
        round_list.update({"prompts": [], "usage": []})

    def _unpack_prompts(self, prompt, round_list: dict[str, list]):
        if isinstance(prompt, (list, tuple)):
            for p in prompt:
                self._unpack_prompts(p, round_list)

        elif isinstance(prompt, Prompts):
            temp_list: list[BasePrompt | TokenUsage] = prompt.prompts.copy()
            if prompt.usage is not None:
                temp_list.append(prompt.usage)

            self._unpack_prompts(temp_list, round_list)

        elif isinstance(prompt, SystemPrompt):
            self._system_prompt = prompt
            if self._messages and self._messages[0].role == "system":
                self._messages[0] = prompt

            else:
                self._messages.insert(0, prompt)

        elif isinstance(prompt, ToolInput):
            if prompt not in self._tools:
                self._tools.append(prompt)

        elif isinstance(prompt, BasePrompt):
            self._messages.append(prompt)
            if prompt.role != "assistant":
                if (
                    round_list["prompts"]
                    and round_list["prompts"][-1].role == "assistant"
                ):
                    self._submit_round(round_list)

            round_list["prompts"].append(prompt)

        elif isinstance(prompt, TokenUsage):
            self._add_usage(prompt)
            round_list["usage"].append(prompt)

        else:
            raise TypeError(f"不支持{type(prompt)}类型创建Context")

    def append(self, addition: Prompts | list | BasePrompt) -> None:
        if not addition:
            return

        if self._rounds:
            key = next(reversed(self._rounds))
            last_round = {"prompts": self._rounds[key].prompts.copy(), "usage": []}
            del self._rounds[key]

        else:
            last_round = {"prompts": [], "usage": []}

        self._unpack_prompts(addition, last_round)
        if last_round.get("prompts", []):
            self._submit_round(last_round)

    def extend(self, addition: Prompts | list | BasePrompt) -> None:
        return self.append(addition)

    def copy(self):
        return self.__copy__()

    def clear(self, system: bool = False, tools: bool = False):
        self._usage_accumulation.clear()
        self._rounds.clear()
        self._messages.clear()

        if system:
            self._system_prompt = None

        else:
            if self._system_prompt is not None:
                self._messages.append(self._system_prompt)

        if tools:
            self._tools.clear()

    def to_prompts(self) -> Prompts:
        return Prompts(self._messages, self._tools)

    def discard_prompt_ids(self, prompt_ids: set[str]) -> int:
        if not prompt_ids:
            return 0

        original_count = len(self._messages)
        self._messages = [
            prompt for prompt in self._messages if prompt.id not in prompt_ids
        ]

        for round_id, prompts in list(self._rounds.items()):
            filtered_prompts = prompts.copy()
            filtered_prompts.prompts = [
                prompt
                for prompt in filtered_prompts.prompts
                if prompt.id not in prompt_ids
            ]
            if filtered_prompts:
                self._rounds[round_id] = filtered_prompts

            else:
                self._rounds.pop(round_id)

        return original_count - len(self._messages)

    def back2last_round(self) -> Prompts:
        """回滚到上一轮对话，POP当前轮"""
        if not self._rounds:
            raise ContextEmptyError("已无上一轮")

        last_round = self._rounds.pop(next(reversed(self._rounds)))
        self._messages.clear()
        if self._system_prompt is not None:
            self._messages.append(self._system_prompt)

        if self._rounds:
            for prompts in self._rounds.values():
                for prompt in prompts:
                    if not isinstance(prompt, SystemPrompt):
                        self._messages.append(prompt)

        return last_round

    def to_dict(self):
        messages = {prompt.id: prompt.to_dict() for prompt in self._messages.copy()}
        tools = {prompt.id: prompt.to_dict() for prompt in self._tools.copy()}
        system = self._system_prompt.id if self._system_prompt is not None else ""
        usage = self._usage_accumulation.to_dict()
        rounds = {}
        for r, data in self._rounds.copy().items():
            round_data = {
                "messages": [p.id for p in data.prompts],
                "usage": data.usage.to_dict() if data.usage is not None else {},
            }
            rounds[r] = round_data

        dump_content = {
            "communication_session": self.session_id,
            "system": system,
            "messages": messages,
            "round": rounds,
            "tools": tools,
            "usage": usage,
        }
        return dump_content

    @classmethod
    def from_dict(cls, data: dict, function_mapping: dict):
        new_context = Context()
        new_context.session_id = data.get("communication_session") or uuid4().hex
        system = data.get("system", "")
        messages = data.get("messages", {})
        rounds = data.get("round", {})
        tools = data.get("tools", {})
        usage = data.get("usage", {})

        parsed_prompts = {}
        parsed_tools = {}
        parsed_rounds = {}
        # 获取完整prompt对象
        for mid, message in messages.items():
            if PromptType(message["type"]) is PromptType.SYSTEM:
                parsed_prompts[mid] = SystemPrompt.from_dict(message)

            elif PromptType(message["type"]) is PromptType.TEXT:
                parsed_prompts[mid] = TextPrompt.from_dict(message)

            elif PromptType(message["type"]) is PromptType.IMAGE:
                parsed_prompts[mid] = ImagePrompt.from_dict(message)

            elif PromptType(message["type"]) is PromptType.TOOL_CALL:
                parsed_prompts[mid] = ToolCall.from_dict(message)

            elif PromptType(message["type"]) is PromptType.TOOL_RESP:
                parsed_prompts[mid] = ToolResponse.from_dict(message)

            if mid in parsed_prompts:
                parsed_prompts[mid].native = copy.deepcopy(message.get("native", {}))

        # 获取tool对象
        for tid, tool in tools.items():
            if PromptType(tool["type"]) is not PromptType.TOOL_IN:
                continue

            function_name = tool["data"]["function_name"]
            function = function_mapping.get(function_name, None)
            if function is None:
                continue

            tool["data"]["function"] = function
            parsed_tools[tid] = ToolInput.from_dict(tool)

        # 重构rounds
        for r, data in rounds.items():
            round_prompts = Prompts()
            for mid in data["messages"]:
                if mid not in parsed_prompts:
                    continue

                round_prompts.append(parsed_prompts[mid])

            if data["usage"]:
                round_prompts.usage = TokenUsage.from_dict(data["usage"])

            if round_prompts:
                parsed_rounds[r] = round_prompts

        new_context._messages.extend(list(parsed_prompts.values()))
        new_context._rounds.update(parsed_rounds)
        new_context._tools.extend(list(parsed_tools.values()))
        if system and system in parsed_prompts:
            new_context._system_prompt = parsed_prompts[system]

        if usage:
            new_context._usage_accumulation = TokenUsage.from_dict(usage)

        return new_context

    @property
    def last_round(self) -> Prompts:
        if not self._rounds:
            return Prompts(role="ROUND")

        last_key = next(reversed(self._rounds))
        last_round = self._rounds[last_key]
        return last_round.copy()

    @property
    def tools(self) -> Prompts:
        """
        返回输入工具列表
        :return:
        """
        return Prompts(self._tools)

    @property
    def rounds(self) -> dict[str, Prompts]:
        return self._rounds.copy()

    @property
    def usage(self) -> TokenUsage | None:
        """返回累积token消耗"""
        return self._usage_accumulation.copy()

    @property
    def system(self) -> SystemPrompt | None:
        if self._system_prompt is None:
            return None

        if not self._system_prompt.text:
            return None

        return self._system_prompt.copy()

    @property
    def has_content(self) -> bool:
        if self._system_prompt is not None:
            return len(self._messages) > 1

        else:
            return bool(self._messages)

    def __str__(self):
        output = ""
        for prompt in self._messages:
            output += str(prompt)

        for prompt in self._tools:
            output += str(prompt)

        return output

    def __copy__(self):
        result = copy.deepcopy(self)
        result.session_id = uuid4().hex
        return result

    def __iter__(self):
        iter_list = self._messages + self._tools
        return iter(iter_list)

    def __reversed__(self):
        iter_list = self._messages + self._tools
        return reversed(iter_list)

    def __bool__(self):
        return bool(self._messages)


@dataclass(slots=True)
class CallOutcome:
    status: str = "idle"
    retry_allowed: bool = False
    call_id: str = ""
    request_id: str = ""
    remote_state: str = "unknown"
    category: str = ""
    committed: bool = False


@dataclass(slots=True)
class AskTask:
    """
    装饰请求任务
    """

    prompts: Prompts
    timeout: float
    id: str = field(default_factory=lambda: str(time.time_ns()))
    event: asyncio.Event = field(default_factory=asyncio.Event)
    response: None | Prompts = None
    handle: asyncio.Task | None = None
    pending: Any = None
    error: BaseException | None = None
    cancelled: bool = False
    committed: bool = False
    outcome: CallOutcome | None = None


@dataclass(frozen=True)
class MediaRef:
    kind: str  # cache/path/url/bytes/base64/remote
    value: Any
    mime_type: str | None = None
    scope: str | None = None
    width: int | None = None
    height: int | None = None
    expires_at: float | None = None


@dataclass
class ImageGenerateRequest:
    model: str
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    background: str | None = None
    output_format: str | None = None
    output_compression: int | None = None
    stream: bool | None = None
    partial_images: int | None = None
    params: dict = field(default_factory=dict)


@dataclass
class ImageEditRequest(ImageGenerateRequest):
    images: list[MediaRef] = field(default_factory=list)
    mask: MediaRef | None = None


@dataclass
class ImageResponseItem:
    image: MediaRef
    revised_prompt: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ImagesResponse:
    created: int | None = None
    data: list[ImageResponseItem] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    request_id: str = ""
    metadata: dict = field(default_factory=dict)
